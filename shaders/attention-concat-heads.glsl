#version 450

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Multi-head output (batch, seq_len, num_heads, head_dim)
layout(set = 0, binding = 0) readonly buffer MultiHeadOutput {
    float mh_output[];
};

// Concatenated output (batch, seq_len, num_heads * head_dim)
layout(set = 0, binding = 1) buffer ConcatenatedOutput {
    float concat_output[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint batch_size;
    uint seq_len;
    uint num_heads;
    uint head_dim;
};

void main() {
    uint gID = gl_GlobalInvocationID.x;
    uint total_elements = batch_size * seq_len * num_heads * head_dim;
    
    if (gID >= total_elements) {
        return;
    }
    
    // Decode position: [batch, seq, head, dim]
    uint batch_idx = gID / (seq_len * num_heads * head_dim);
    uint remainder = gID % (seq_len * num_heads * head_dim);
    uint seq_idx = remainder / (num_heads * head_dim);
    uint head_dim_remainder = remainder % (num_heads * head_dim);
    uint head_idx = head_dim_remainder / head_dim;
    uint dim_idx = head_dim_remainder % head_dim;
    
    // Input layout: [batch, seq, head, dim]
    uint in_idx = batch_idx * seq_len * num_heads * head_dim +
                  seq_idx * num_heads * head_dim +
                  head_idx * head_dim + dim_idx;
    
    // Output layout: [batch, seq, head * dim] (concatenated)
    uint out_idx = batch_idx * seq_len * num_heads * head_dim +
                   seq_idx * num_heads * head_dim +
                   head_idx * head_dim + dim_idx;
    
    concat_output[out_idx] = mh_output[in_idx];
}
