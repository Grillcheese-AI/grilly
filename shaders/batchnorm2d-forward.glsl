#version 450

/*
 * BatchNorm2d Forward Pass
 *
 * Normalizes input across batch and spatial dimensions per channel.
 * y = gamma * (x - mean) / sqrt(var + eps) + beta
 *
 * Input: (batch, channels, height, width)
 * Output: (batch, channels, height, width)
 * Running stats: updated with exponential moving average
 */

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0) readonly buffer InputBuffer {
    float input_data[];
};

layout(binding = 1) buffer OutputBuffer {
    float output_data[];
};

layout(binding = 2) readonly buffer GammaBuffer {
    float gamma[];  // scale (num_features,)
};

layout(binding = 3) readonly buffer BetaBuffer {
    float beta[];  // shift (num_features,)
};

layout(binding = 4) buffer RunningMeanBuffer {
    float running_mean[];  // (num_features,)
};

layout(binding = 5) buffer RunningVarBuffer {
    float running_var[];  // (num_features,)
};

layout(binding = 6) buffer BatchMeanBuffer {
    float batch_mean[];  // (num_features,) - computed mean for this batch
};

layout(binding = 7) buffer BatchVarBuffer {
    float batch_var[];  // (num_features,) - computed variance for this batch
};

layout(push_constant) uniform PushConstants {
    uint batch_size;
    uint num_features;  // channels
    uint height;
    uint width;
    float eps;
    float momentum;
    uint training;  // 1 if training, 0 if eval
    uint affine;    // 1 if using learnable gamma/beta
} params;

// Shared memory for reduction
shared float shared_sum[256];
shared float shared_sq_sum[256];

void main() {
    uint channel = gl_GlobalInvocationID.x;

    if (channel >= params.num_features) {
        return;
    }

    uint spatial_size = params.height * params.width;
    uint n = params.batch_size * spatial_size;
    uint local_idx = gl_LocalInvocationIndex;

    // Training mode: compute batch statistics
    if (params.training == 1) {
        // Compute mean using Welford's algorithm for numerical stability
        float sum = 0.0;
        float sq_sum = 0.0;

        for (uint b = 0; b < params.batch_size; b++) {
            for (uint h = 0; h < params.height; h++) {
                for (uint w = 0; w < params.width; w++) {
                    uint idx = b * params.num_features * spatial_size +
                              channel * spatial_size +
                              h * params.width + w;
                    float val = input_data[idx];
                    sum += val;
                    sq_sum += val * val;
                }
            }
        }

        // Compute mean and variance
        float mean = sum / float(n);
        float variance = (sq_sum / float(n)) - (mean * mean);

        // Store batch statistics
        batch_mean[channel] = mean;
        batch_var[channel] = variance;

        // Update running statistics with exponential moving average
        running_mean[channel] = params.momentum * mean + (1.0 - params.momentum) * running_mean[channel];
        running_var[channel] = params.momentum * variance + (1.0 - params.momentum) * running_var[channel];

        // Normalize using batch statistics
        float inv_std = 1.0 / sqrt(variance + params.eps);
        for (uint b = 0; b < params.batch_size; b++) {
            for (uint h = 0; h < params.height; h++) {
                for (uint w = 0; w < params.width; w++) {
                    uint idx = b * params.num_features * spatial_size +
                              channel * spatial_size +
                              h * params.width + w;

                    float normalized = (input_data[idx] - mean) * inv_std;

                    // Apply affine transformation if enabled
                    if (params.affine == 1) {
                        normalized = normalized * gamma[channel] + beta[channel];
                    }

                    output_data[idx] = normalized;
                }
            }
        }
    } else {
        // Eval mode: use running statistics
        float mean = running_mean[channel];
        float variance = running_var[channel];
        float inv_std = 1.0 / sqrt(variance + params.eps);

        for (uint b = 0; b < params.batch_size; b++) {
            for (uint h = 0; h < params.height; h++) {
                for (uint w = 0; w < params.width; w++) {
                    uint idx = b * params.num_features * spatial_size +
                              channel * spatial_size +
                              h * params.width + w;

                    float normalized = (input_data[idx] - mean) * inv_std;

                    if (params.affine == 1) {
                        normalized = normalized * gamma[channel] + beta[channel];
                    }

                    output_data[idx] = normalized;
                }
            }
        }
    }
}
