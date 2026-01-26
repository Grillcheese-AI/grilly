#version 450

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Attention scores (batch, num_heads, seq_len, seq_len)
layout(set = 0, binding = 0) buffer AttentionScores {
    float scores[];
};

// Attention mask buffer
// For causal: (seq_len, seq_len) - 1 = allow, 0 = mask
// For custom: (batch, seq_len) - 1 = allow, 0 = mask (applied to k_pos dimension)
layout(set = 0, binding = 1) readonly buffer AttentionMask {
    float mask[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint batch_size;
    uint num_heads;
    uint seq_len;
    uint use_causal_mask;  // 1 = apply causal mask, 0 = use custom mask from buffer
    float mask_value;      // Large negative value (e.g., -1e9)
};

void main() {
    uint gID = gl_GlobalInvocationID.x;
    uint total_scores = batch_size * num_heads * seq_len * seq_len;
    
    if (gID >= total_scores) {
        return;
    }
    
    // Decode position
    uint batch_idx = gID / (num_heads * seq_len * seq_len);
    uint remainder = gID % (num_heads * seq_len * seq_len);
    uint head_idx = remainder / (seq_len * seq_len);
    uint pos_remainder = remainder % (seq_len * seq_len);
    uint q_pos = pos_remainder / seq_len;
    uint k_pos = pos_remainder % seq_len;
    
    if (use_causal_mask == 1) {
        // Causal mask: can only attend to previous positions
        if (k_pos > q_pos) {
            scores[gID] = mask_value;
        }
    } else {
        // Custom mask: use mask buffer
        // Mask shape: (batch, seq_len) - mask out positions where mask[batch_idx][k_pos] == 0
        uint mask_idx = batch_idx * seq_len + k_pos;
        if (mask[mask_idx] < 0.5) {  // Mask out if mask value is < 0.5
            scores[gID] = mask_value;
        }
    }
}
