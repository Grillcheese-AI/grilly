#version 450

/*
 * Fused LayerNorm + Linear Shader
 *
 * Combines layer normalization and linear transformation.
 * Common pattern in pre-norm transformers: Linear(LayerNorm(x))
 *
 * Multi-pass operation:
 * - Pass 0: Compute mean across feature dimension
 * - Pass 1: Compute variance
 * - Pass 2: Normalize and apply linear transformation (fused)
 */

layout (local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

// Input tensor (batch * seq, features)
layout(set = 0, binding = 0) readonly buffer Input {
    float input_data[];
};

// Gamma (scale) parameters for LayerNorm (features)
layout(set = 0, binding = 1) readonly buffer Gamma {
    float gamma[];
};

// Beta (shift) parameters for LayerNorm (features)
layout(set = 0, binding = 2) readonly buffer Beta {
    float beta[];
};

// Linear weight matrix (output_dim, features)
layout(set = 0, binding = 3) readonly buffer Weights {
    float W[];
};

// Linear bias vector (output_dim)
layout(set = 0, binding = 4) readonly buffer Bias {
    float b[];
};

// Output (batch * seq, output_dim)
layout(set = 0, binding = 5) buffer Output {
    float output_data[];
};

// Mean buffer (batch * seq)
layout(set = 0, binding = 6) buffer MeanBuffer {
    float mean_vals[];
};

// Variance buffer (batch * seq)
layout(set = 0, binding = 7) buffer VarianceBuffer {
    float var_vals[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint batch_seq;       // batch_size * seq_len
    uint input_dim;       // features (for LayerNorm)
    uint output_dim;      // output dimension (for Linear)
    float eps;            // LayerNorm epsilon
    uint pass_type;       // 0 = mean, 1 = variance, 2 = normalize + linear
    uint has_bias;        // 1 if linear bias exists
};

void main() {
    if (pass_type == 0) {
        // Pass 1: Compute mean along feature dimension
        uint pos = gl_GlobalInvocationID.x;
        if (pos >= batch_seq) return;

        float sum = 0.0;
        for (uint f = 0; f < input_dim; f++) {
            sum += input_data[pos * input_dim + f];
        }
        mean_vals[pos] = sum / float(input_dim);

    } else if (pass_type == 1) {
        // Pass 2: Compute variance
        uint pos = gl_GlobalInvocationID.x;
        if (pos >= batch_seq) return;

        float mean = mean_vals[pos];
        float sum_sq = 0.0;
        for (uint f = 0; f < input_dim; f++) {
            float diff = input_data[pos * input_dim + f] - mean;
            sum_sq += diff * diff;
        }
        var_vals[pos] = sum_sq / float(input_dim);

    } else if (pass_type == 2) {
        // Pass 3: Fused Normalize + Linear
        // Each thread computes one output element
        uint row = gl_GlobalInvocationID.y;  // Sample index
        uint col = gl_GlobalInvocationID.x;  // Output feature index

        if (row >= batch_seq || col >= output_dim) {
            return;
        }

        // Get mean and std for this position
        float mean = mean_vals[row];
        float variance = var_vals[row];
        float inv_std = 1.0 / sqrt(variance + eps);

        // Compute: output = Linear(LayerNorm(x))
        // = sum_k[ W[col][k] * (gamma[k] * (x[k] - mean) / std + beta[k]) ]
        float sum = 0.0;

        for (uint k = 0; k < input_dim; k++) {
            // Normalize input
            float x = input_data[row * input_dim + k];
            float normalized = (x - mean) * inv_std;
            float ln_out = gamma[k] * normalized + beta[k];

            // Apply linear weight
            sum += W[col * input_dim + k] * ln_out;
        }

        // Add linear bias if present
        if (has_bias == 1) {
            sum += b[col];
        }

        output_data[row * output_dim + col] = sum;
    }
}
