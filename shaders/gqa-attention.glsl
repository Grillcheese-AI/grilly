#version 450

// GQA Decode Attention: single-token query against KV-cache
// Fuses repeat_kv: maps query heads to KV heads via integer division
// kv_head_idx = q_head_idx / (num_q_heads / num_kv_heads)

layout (local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

// Query: (batch, 1, num_q_heads, head_dim) - single decode token
layout(set = 0, binding = 0) readonly buffer Queries {
    float Q[];
};

// Cached Keys: (batch, cache_len, num_kv_heads, head_dim)
layout(set = 0, binding = 1) readonly buffer KeyCache {
    float K_cache[];
};

// Cached Values: (batch, cache_len, num_kv_heads, head_dim)
layout(set = 0, binding = 2) readonly buffer ValueCache {
    float V_cache[];
};

// Output: (batch, 1, num_q_heads, head_dim)
layout(set = 0, binding = 3) buffer Output {
    float output_data[];
};

// Softmax scratch: (batch, num_q_heads, cache_len)
layout(set = 0, binding = 4) buffer ScratchScores {
    float scores[];
};

layout(push_constant) uniform PushConsts {
    uint batch_size;
    uint num_q_heads;
    uint num_kv_heads;
    uint head_dim;
    uint cache_len;
    float scale;           // 1.0 / sqrt(head_dim)
};

void main() {
    uint col = gl_GlobalInvocationID.x;
    uint row = gl_GlobalInvocationID.y;

    // row encodes (batch, q_head), col encodes cache position or head_dim
    uint total_q = batch_size * num_q_heads;

    // Phase 1: Compute attention scores Q @ K^T (each thread: one score)
    // row = batch*num_q_heads + q_head, col = cache_pos
    if (row < total_q && col < cache_len) {
        uint batch_idx = row / num_q_heads;
        uint q_head = row % num_q_heads;

        // GQA mapping: which KV head does this query head attend to?
        uint kv_group_size = num_q_heads / num_kv_heads;
        uint kv_head = q_head / kv_group_size;

        // Compute dot product: Q[batch, 0, q_head, :] @ K[batch, col, kv_head, :]
        float dot = 0.0;
        for (uint d = 0; d < head_dim; d++) {
            uint q_idx = batch_idx * num_q_heads * head_dim + q_head * head_dim + d;
            uint k_idx = batch_idx * cache_len * num_kv_heads * head_dim
                       + col * num_kv_heads * head_dim
                       + kv_head * head_dim + d;
            dot += Q[q_idx] * K_cache[k_idx];
        }

        uint score_idx = batch_idx * num_q_heads * cache_len + q_head * cache_len + col;
        scores[score_idx] = dot * scale;
    }

    barrier();
    memoryBarrierBuffer();

    // Phase 2: Softmax over cache dimension (one thread per q_head)
    if (row < total_q && col == 0) {
        uint batch_idx = row / num_q_heads;
        uint q_head = row % num_q_heads;
        uint base = batch_idx * num_q_heads * cache_len + q_head * cache_len;

        // Find max for numerical stability
        float max_val = -1e10;
        for (uint c = 0; c < cache_len; c++) {
            float s = scores[base + c];
            if (s > max_val) max_val = s;
        }

        // Compute exp and sum
        float sum_exp = 0.0;
        for (uint c = 0; c < cache_len; c++) {
            float e = exp(scores[base + c] - max_val);
            scores[base + c] = e;
            sum_exp += e;
        }

        // Normalize
        float inv_sum = 1.0 / max(sum_exp, 1e-10);
        for (uint c = 0; c < cache_len; c++) {
            scores[base + c] *= inv_sum;
        }
    }

    barrier();
    memoryBarrierBuffer();

    // Phase 3: Weighted sum: output = scores @ V_cache
    // row = batch*num_q_heads + q_head, col = head_dim index
    if (row < total_q && col < head_dim) {
        uint batch_idx = row / num_q_heads;
        uint q_head = row % num_q_heads;
        uint kv_group_size = num_q_heads / num_kv_heads;
        uint kv_head = q_head / kv_group_size;

        uint score_base = batch_idx * num_q_heads * cache_len + q_head * cache_len;

        float sum = 0.0;
        for (uint c = 0; c < cache_len; c++) {
            float w = scores[score_base + c];
            uint v_idx = batch_idx * cache_len * num_kv_heads * head_dim
                       + c * num_kv_heads * head_dim
                       + kv_head * head_dim + col;
            sum += w * V_cache[v_idx];
        }

        uint out_idx = batch_idx * num_q_heads * head_dim + q_head * head_dim + col;
        output_data[out_idx] = sum;
    }
}
