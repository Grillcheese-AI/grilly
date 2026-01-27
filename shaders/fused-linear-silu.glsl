#version 450

/*
 * Fused Linear + SiLU/Swish Shader
 *
 * Combines linear transformation and SiLU activation in a single pass.
 * SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
 *
 * Common in LLaMA, Mistral, and other modern architectures.
 */

layout (local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

// Input features (batch * seq, input_dim)
layout(set = 0, binding = 0) readonly buffer Input {
    float input_data[];
};

// Weight matrix (output_dim, input_dim)
layout(set = 0, binding = 1) readonly buffer Weights {
    float W[];
};

// Bias vector (output_dim)
layout(set = 0, binding = 2) readonly buffer Bias {
    float b[];
};

// Output (batch * seq, output_dim)
layout(set = 0, binding = 3) buffer Output {
    float output_data[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint batch_seq;       // batch_size * seq_len
    uint input_dim;
    uint output_dim;
    uint has_bias;        // 1 if bias exists, 0 otherwise
};

// SiLU/Swish: x * sigmoid(x)
float silu(float x) {
    // Clamp to prevent exp overflow
    float x_clamped = clamp(x, -88.0, 88.0);
    float sigmoid_x = 1.0 / (1.0 + exp(-x_clamped));
    return x * sigmoid_x;
}

void main() {
    uint row = gl_GlobalInvocationID.y;  // Sample index
    uint col = gl_GlobalInvocationID.x;  // Output feature index

    if (row >= batch_seq || col >= output_dim) {
        return;
    }

    // Step 1: Linear transformation
    float sum = 0.0;

    for (uint k = 0; k < input_dim; k++) {
        uint input_idx = row * input_dim + k;
        uint weight_idx = col * input_dim + k;
        sum += input_data[input_idx] * W[weight_idx];
    }

    // Add bias if present
    if (has_bias == 1) {
        sum += b[col];
    }

    // Step 2: Apply SiLU activation (fused)
    float silu_out = silu(sum);

    // Write final result
    uint out_idx = row * output_dim + col;
    output_data[out_idx] = silu_out;
}
