#version 450

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Cross-attention weights (batch, num_heads, decoder_len, encoder_len)
// T5 uses encoder-decoder architecture with cross-attention
layout(set = 0, binding = 0) readonly buffer AttentionWeights {
    float weights[];
};

// Encoder values (batch, encoder_len, num_heads, head_dim)
layout(set = 0, binding = 1) readonly buffer EncoderValues {
    float V[];
};

// Output (batch, decoder_len, num_heads, head_dim)
layout(set = 0, binding = 2) buffer Output {
    float output_data[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint batch_size;
    uint decoder_len;  // Decoder sequence length
    uint encoder_len;  // Encoder sequence length
    uint num_heads;
    uint head_dim;
};

void main() {
    // Each thread computes one output element
    uint gID = gl_GlobalInvocationID.x;
    uint total_elements = batch_size * decoder_len * num_heads * head_dim;
    
    if (gID >= total_elements) {
        return;
    }
    
    // Decode position from linear index
    uint remainder = gID;
    uint batch_idx = remainder / (decoder_len * num_heads * head_dim);
    remainder = remainder % (decoder_len * num_heads * head_dim);
    uint seq_q = remainder / (num_heads * head_dim);  // Decoder position
    remainder = remainder % (num_heads * head_dim);
    uint head_idx = remainder / head_dim;
    uint dim_idx = remainder % head_dim;
    
    // Cross-attention: decoder queries attend to encoder keys/values
    // output[batch, decoder_q, head, dim] = sum_k(weights[batch, head, decoder_q, encoder_k] * V[batch, encoder_k, head, dim])
    float sum = 0.0;
    
    for (uint k = 0; k < encoder_len; k++) {
        // Weight index: weights[batch, head, decoder_q, encoder_k]
        uint weight_idx = batch_idx * num_heads * decoder_len * encoder_len +
                         head_idx * decoder_len * encoder_len +
                         seq_q * encoder_len + k;
        
        // Encoder value index: V[batch, encoder_k, head, dim]
        uint v_idx = batch_idx * encoder_len * num_heads * head_dim +
                    k * num_heads * head_dim +
                    head_idx * head_dim + dim_idx;
        
        sum += weights[weight_idx] * V[v_idx];
    }
    
    // Output index: output[batch, decoder_q, head, dim]
    uint out_idx = batch_idx * decoder_len * num_heads * head_dim +
                   seq_q * num_heads * head_dim +
                   head_idx * head_dim + dim_idx;
    
    output_data[out_idx] = sum;
}
