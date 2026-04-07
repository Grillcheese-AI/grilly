#version 450

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Attention weights (batch, num_heads, seq_len, seq_len) - softmax already applied
layout(set = 0, binding = 0) readonly buffer AttentionWeights {
    float weights[];
};

// Values (batch, seq_len, num_heads, head_dim)
layout(set = 0, binding = 1) readonly buffer Values {
    float V[];
};

// Output (batch, seq_len, num_heads, head_dim)
layout(set = 0, binding = 2) buffer Output {
    float output_data[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint batch_size;
    uint seq_len;
    uint num_heads;
    uint head_dim;
};

void main() {
    // Each thread computes one output element
    // Total output elements: batch_size * seq_len * num_heads * head_dim
    uint gID = gl_GlobalInvocationID.x;
    uint total_elements = batch_size * seq_len * num_heads * head_dim;
    
    if (gID >= total_elements) {
        return;
    }
    
    // Decode position from linear index
    uint remainder = gID;
    uint batch_idx = remainder / (seq_len * num_heads * head_dim);
    remainder = remainder % (seq_len * num_heads * head_dim);
    uint seq_q = remainder / (num_heads * head_dim);
    remainder = remainder % (num_heads * head_dim);
    uint head_idx = remainder / head_dim;
    uint dim_idx = remainder % head_dim;
    
    // Compute weighted sum: output[batch, seq_q, head, dim]
    //   = sum_k(weights[batch, head, seq_q, k] * V[batch, k, head, dim])
    // Hoist loop-invariant index math and use FMA in the inner loop.
    float sum = 0.0;

    uint weight_base = batch_idx * num_heads * seq_len * seq_len +
                       head_idx * seq_len * seq_len +
                       seq_q * seq_len;

    uint v_base = batch_idx * seq_len * num_heads * head_dim +
                  head_idx * head_dim + dim_idx;
    uint v_stride = num_heads * head_dim;

    for (uint k = 0; k < seq_len; k++) {
        sum = fma(weights[weight_base + k], V[v_base + k * v_stride], sum);
    }
    
    // Output index: output[batch, seq_q, head, dim]
    uint out_idx = batch_idx * seq_len * num_heads * head_dim +
                   seq_q * num_heads * head_dim +
                   head_idx * head_dim + dim_idx;
    
    output_data[out_idx] = sum;
}
