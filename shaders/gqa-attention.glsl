#version 450

// GQA Decode Attention: single-token query against KV-cache
// Fuses repeat_kv: maps query heads to KV heads via integer division
//
// Each workgroup handles ONE (batch, q_head) pair.
// gl_WorkGroupID.x = linear index of (batch * num_q_heads + q_head)
// gl_LocalInvocationID.x = thread within workgroup (for parallel reduction)

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer Queries { float Q[]; };
layout(set = 0, binding = 1) readonly buffer KeyCache { float K_cache[]; };
layout(set = 0, binding = 2) readonly buffer ValueCache { float V_cache[]; };
layout(set = 0, binding = 3) buffer Output { float output_data[]; };
layout(set = 0, binding = 4) buffer ScratchScores { float scores[]; };

layout(push_constant) uniform PushConsts {
    uint batch_size;
    uint num_q_heads;
    uint num_kv_heads;
    uint head_dim;
    uint cache_len;
    float scale;
};

shared float s_max;
shared float s_sum;

void main() {
    uint wg = gl_WorkGroupID.x;        // which (batch, q_head)
    uint tid = gl_LocalInvocationID.x;  // thread within workgroup
    uint wg_size = gl_WorkGroupSize.x;  // 256

    uint total_q = batch_size * num_q_heads;
    if (wg >= total_q) return;

    uint batch_idx = wg / num_q_heads;
    uint q_head = wg % num_q_heads;
    uint kv_group_size = num_q_heads / num_kv_heads;
    uint kv_head = q_head / kv_group_size;

    uint score_base = batch_idx * num_q_heads * cache_len + q_head * cache_len;

    // ── Phase 1: Compute attention scores Q @ K^T ──────────────────────
    // Each thread handles multiple cache positions (stride loop)
    for (uint c = tid; c < cache_len; c += wg_size) {
        float dot = 0.0;
        for (uint d = 0; d < head_dim; d++) {
            uint q_idx = batch_idx * num_q_heads * head_dim + q_head * head_dim + d;
            uint k_idx = batch_idx * cache_len * num_kv_heads * head_dim
                       + c * num_kv_heads * head_dim
                       + kv_head * head_dim + d;
            dot += Q[q_idx] * K_cache[k_idx];
        }
        scores[score_base + c] = dot * scale;
    }

    barrier();
    memoryBarrierBuffer();

    // ── Phase 2: Softmax (single thread finds max + normalizes) ────────
    if (tid == 0) {
        float max_val = -1e10;
        for (uint c = 0; c < cache_len; c++) {
            float s = scores[score_base + c];
            if (s > max_val) max_val = s;
        }
        s_max = max_val;
    }

    barrier();

    // Compute exp (parallel)
    for (uint c = tid; c < cache_len; c += wg_size) {
        scores[score_base + c] = exp(scores[score_base + c] - s_max);
    }

    barrier();
    memoryBarrierBuffer();

    // Sum (single thread)
    if (tid == 0) {
        float sum_exp = 0.0;
        for (uint c = 0; c < cache_len; c++) {
            sum_exp += scores[score_base + c];
        }
        s_sum = max(sum_exp, 1e-10);
    }

    barrier();

    // Normalize (parallel)
    float inv_sum = 1.0 / s_sum;
    for (uint c = tid; c < cache_len; c += wg_size) {
        scores[score_base + c] *= inv_sum;
    }

    barrier();
    memoryBarrierBuffer();

    // ── Phase 3: Weighted sum: output = softmax_scores @ V ─────────────
    // Each thread handles multiple head_dim positions
    for (uint d = tid; d < head_dim; d += wg_size) {
        float sum = 0.0;
        for (uint c = 0; c < cache_len; c++) {
            float w = scores[score_base + c];
            uint v_idx = batch_idx * cache_len * num_kv_heads * head_dim
                       + c * num_kv_heads * head_dim
                       + kv_head * head_dim + d;
            sum += w * V_cache[v_idx];
        }
        uint out_idx = batch_idx * num_q_heads * head_dim + q_head * head_dim + d;
        output_data[out_idx] = sum;
    }
}
