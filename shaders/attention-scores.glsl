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
    uint q_pos = gl_GlobalInvocationID.y;
    uint k_pos = gl_GlobalInvocationID.x;
    uint flat_bh = gl_GlobalInvocationID.z;

    if (q_pos >= seq_len || k_pos >= seq_len || flat_bh >= batch_size * num_heads) {
        return;
    }

    uint batch_idx = flat_bh / num_heads;
    uint head_idx  = flat_bh % num_heads;

    float score = 0.0;

    for (uint d = 0; d < head_dim; d++) {
        uint base = batch_idx * seq_len * num_heads * head_dim
                  + head_idx * head_dim;
        uint q_idx = base + q_pos * num_heads * head_dim + d;
        uint k_idx = base + k_pos * num_heads * head_dim + d;

        score += Q[q_idx] * K[k_idx];
    }

    score *= scale;

    uint score_idx =
        batch_idx * num_heads * seq_len * seq_len +
        head_idx  * seq_len * seq_len +
        q_pos     * seq_len +
        k_pos;

    scores[score_idx] = score;
}
