#version 450

/*
 * BatchNorm2d Backward Pass
 *
 * Computes gradients w.r.t. input, gamma, and beta.
 * Uses batch statistics computed during forward pass.
 *
 * grad_input = gamma / sqrt(var + eps) * (
 *     grad_output
 *     - mean(grad_output)
 *     - (x - mean) / (var + eps) * mean(grad_output * (x - mean))
 * )
 */

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0) readonly buffer GradOutputBuffer {
    float grad_output[];
};

layout(binding = 1) readonly buffer InputBuffer {
    float input_data[];
};

layout(binding = 2) buffer GradInputBuffer {
    float grad_input[];
};

layout(binding = 3) readonly buffer BatchMeanBuffer {
    float batch_mean[];  // From forward pass
};

layout(binding = 4) readonly buffer BatchVarBuffer {
    float batch_var[];  // From forward pass
};

layout(binding = 5) readonly buffer GammaBuffer {
    float gamma[];
};

layout(binding = 6) buffer GradGammaBuffer {
    float grad_gamma[];  // (num_features,)
};

layout(binding = 7) buffer GradBetaBuffer {
    float grad_beta[];  // (num_features,)
};

layout(push_constant) uniform PushConstants {
    uint batch_size;
    uint num_features;
    uint height;
    uint width;
    float eps;
    uint affine;  // 1 if using learnable gamma/beta
} params;

void main() {
    uint channel = gl_GlobalInvocationID.x;

    if (channel >= params.num_features) {
        return;
    }

    uint spatial_size = params.height * params.width;
    uint n = params.batch_size * spatial_size;

    float mean = batch_mean[channel];
    float variance = batch_var[channel];
    float inv_std = 1.0 / sqrt(variance + params.eps);

    // Compute gradient w.r.t. gamma and beta (if affine)
    if (params.affine == 1) {
        float grad_gamma_sum = 0.0;
        float grad_beta_sum = 0.0;

        for (uint b = 0; b < params.batch_size; b++) {
            for (uint h = 0; h < params.height; h++) {
                for (uint w = 0; w < params.width; w++) {
                    uint idx = b * params.num_features * spatial_size +
                              channel * spatial_size +
                              h * params.width + w;

                    float x_normalized = (input_data[idx] - mean) * inv_std;
                    grad_gamma_sum += grad_output[idx] * x_normalized;
                    grad_beta_sum += grad_output[idx];
                }
            }
        }

        grad_gamma[channel] = grad_gamma_sum;
        grad_beta[channel] = grad_beta_sum;
    }

    // Compute intermediate values for input gradient
    float grad_output_sum = 0.0;
    float grad_output_dot_normalized = 0.0;

    for (uint b = 0; b < params.batch_size; b++) {
        for (uint h = 0; h < params.height; h++) {
            for (uint w = 0; w < params.width; w++) {
                uint idx = b * params.num_features * spatial_size +
                          channel * spatial_size +
                          h * params.width + w;

                float x_centered = input_data[idx] - mean;
                float x_normalized = x_centered * inv_std;

                grad_output_sum += grad_output[idx];
                grad_output_dot_normalized += grad_output[idx] * x_normalized;
            }
        }
    }

    float grad_output_mean = grad_output_sum / float(n);
    float grad_output_dot_mean = grad_output_dot_normalized / float(n);

    // Compute gradient w.r.t. input
    float scale = (params.affine == 1) ? gamma[channel] * inv_std : inv_std;

    for (uint b = 0; b < params.batch_size; b++) {
        for (uint h = 0; h < params.height; h++) {
            for (uint w = 0; w < params.width; w++) {
                uint idx = b * params.num_features * spatial_size +
                          channel * spatial_size +
                          h * params.width + w;

                float x_centered = input_data[idx] - mean;
                float x_normalized = x_centered * inv_std;

                // Gradient computation following PyTorch's formula
                float grad = grad_output[idx] - grad_output_mean - x_normalized * grad_output_dot_mean;
                grad_input[idx] = scale * grad;
            }
        }
    }
}
