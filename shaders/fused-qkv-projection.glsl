#version 450

/*
 * Fused QKV Projection Shader
 *
 * Computes Q, K, V projections in a single pass.
 * Instead of 3 separate matmuls, combines weights into [W_q; W_k; W_v].
 *
 * Input: x (batch, seq_len, hidden_dim)
 * Output: QKV (batch, seq_len, 3 * num_heads * head_dim)
 *
 * Common pattern in transformer attention layers.
 */

layout (local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

// Input features (batch * seq, hidden_dim)
layout(set = 0, binding = 0) readonly buffer Input {
    float input_data[];
};

// Combined QKV weight matrix (3 * num_heads * head_dim, hidden_dim)
// Packed as: [W_q; W_k; W_v]
layout(set = 0, binding = 1) readonly buffer QKVWeights {
    float W_qkv[];
};

// Combined QKV bias (3 * num_heads * head_dim)
// Packed as: [b_q; b_k; b_v]
layout(set = 0, binding = 2) readonly buffer QKVBias {
    float b_qkv[];
};

// Output QKV (batch * seq, 3 * num_heads * head_dim)
layout(set = 0, binding = 3) buffer Output {
    float output_data[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint batch_seq;       // batch_size * seq_len
    uint hidden_dim;      // input hidden dimension
    uint num_heads;
    uint head_dim;
    uint has_bias;        // 1 if bias exists
};

void main() {
    uint row = gl_GlobalInvocationID.y;  // Sample index (batch * seq)
    uint col = gl_GlobalInvocationID.x;  // Output feature index

    uint qkv_dim = 3 * num_heads * head_dim;

    if (row >= batch_seq || col >= qkv_dim) {
        return;
    }

    // Compute: output[row][col] = sum(input[row][k] * W_qkv[col][k]) + b_qkv[col]
    float sum = 0.0;

    for (uint k = 0; k < hidden_dim; k++) {
        uint input_idx = row * hidden_dim + k;
        uint weight_idx = col * hidden_dim + k;
        sum += input_data[input_idx] * W_qkv[weight_idx];
    }

    // Add bias if present
    if (has_bias == 1) {
        sum += b_qkv[col];
    }

    uint out_idx = row * qkv_dim + col;
    output_data[out_idx] = sum;
}
