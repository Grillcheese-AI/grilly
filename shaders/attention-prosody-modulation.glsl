#version 450

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Attention scores (batch, num_heads, seq_len, seq_len)
layout(set = 0, binding = 0) buffer AttentionScores {
    float scores[];
};

// Prosody features (batch, seq_len, prosody_dim)
layout(set = 0, binding = 1) readonly buffer ProsodyFeatures {
    float prosody[];
};

// Prosody projection weights (num_heads, prosody_dim)
layout(set = 0, binding = 2) readonly buffer ProsodyWeights {
    float W_prosody[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint batch_size;
    uint num_heads;
    uint seq_len;
    uint prosody_dim;
    float prosody_strength;  // Modulation strength (e.g., 0.3)
};

void main() {
    uint gID = gl_GlobalInvocationID.x;
    uint total_scores = batch_size * num_heads * seq_len * seq_len;
    
    if (gID >= total_scores) {
        return;
    }
    
    // Decode indices
    uint batch_idx = gID / (num_heads * seq_len * seq_len);
    uint remainder = gID % (num_heads * seq_len * seq_len);
    uint head_idx = remainder / (seq_len * seq_len);
    uint pos_remainder = remainder % (seq_len * seq_len);
    uint q_pos = pos_remainder / seq_len;
    uint k_pos = pos_remainder % seq_len;
    
    // Compute prosody bias from key position
    float prosody_bias = 0.0;
    
    for (uint d = 0; d < prosody_dim; d++) {
        uint prosody_idx = batch_idx * seq_len * prosody_dim + k_pos * prosody_dim + d;
        uint weight_idx = head_idx * prosody_dim + d;
        
        prosody_bias += prosody[prosody_idx] * W_prosody[weight_idx];
    }
    
    // Apply prosody modulation: score = score + strength * prosody_bias
    scores[gID] += prosody_strength * prosody_bias;
}
