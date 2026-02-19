#version 450

/*
 * Fused Linear + Tanh Shader
 *
 * Combines linear transformation and tanh activation in a single pass.
 * Avoids intermediate global memory write/read for the activation.
 *
 * Used by VSA Reasoning Head: tanh(x @ W.T + b) projects to bipolar space.
 */

layout (local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

// Input features (batch_seq, input_dim)
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

// Output (batch_seq, output_dim)
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

void main() {
    uint row = gl_GlobalInvocationID.y;  // Sample index
    uint col = gl_GlobalInvocationID.x;  // Output feature index

    if (row >= batch_seq || col >= output_dim) {
        return;
    }

    // Step 1: Linear transformation
    // Compute: linear_out = sum(input[row][k] * W[col][k]) + b[col]
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

    // Step 2: Apply tanh activation (fused - no intermediate memory access)
    uint out_idx = row * output_dim + col;
    output_data[out_idx] = tanh(sum);
}
