#version 450

layout (local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

// Query, Key, Value inputs (batch, seq_len, dim)
layout(set = 0, binding = 0) readonly buffer Queries {
    float Q[];
};

layout(set = 0, binding = 1) readonly buffer Keys {
    float K[];
};

layout(set = 0, binding = 2) readonly buffer Values {
    float V[];
};

// Attention scores output (batch, num_heads, seq_len, seq_len)
layout(set = 0, binding = 3) buffer AttentionScores {
    float scores[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint batch_size;
    uint seq_len;
    uint num_heads;
    uint head_dim;
    float scale;         // 1 / sqrt(head_dim)
    uint pass_type;      // 0 = compute scores, 1 = apply mask (optional)
};

void main() {
    uint row = gl_GlobalInvocationID.y;
    uint col = gl_GlobalInvocationID.x;
    
    if (pass_type == 0) {
        // Compute attention scores: scores = Q @ K^T / sqrt(d_k)
        // Each thread computes score[batch][head][q_pos][k_pos]
        
        uint total_positions = batch_size * num_heads * seq_len;
        uint flat_row = row;
        
        if (flat_row >= total_positions || col >= seq_len) {
            return;
        }
        
        uint batch_idx = flat_row / (num_heads * seq_len);
        uint remainder = flat_row % (num_heads * seq_len);
        uint head_idx = remainder / seq_len;
        uint q_pos = remainder % seq_len;
        uint k_pos = col;
        
        // Compute dot product: Q[q_pos] · K[k_pos]
        float score = 0.0;
        
        for (uint d = 0; d < head_dim; d++) {
            // Q index: [batch, seq, head, head_dim]
            uint q_idx = batch_idx * seq_len * num_heads * head_dim +
                        q_pos * num_heads * head_dim +
                        head_idx * head_dim + d;
            
            // K index: [batch, seq, head, head_dim]
            uint k_idx = batch_idx * seq_len * num_heads * head_dim +
                        k_pos * num_heads * head_dim +
                        head_idx * head_dim + d;
            
            score += Q[q_idx] * K[k_idx];
        }
        
        // Scale by sqrt(d_k)
        score *= scale;
        
        // Store score
        uint score_idx = batch_idx * num_heads * seq_len * seq_len +
                        head_idx * seq_len * seq_len +
                        q_pos * seq_len + k_pos;
        
        scores[score_idx] = score;
    }
}
