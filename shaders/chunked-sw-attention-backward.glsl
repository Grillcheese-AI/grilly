#version 450
// Sliding-window causal attention backward (0.0.2) — single-pass, GPU-side.
//
// Recomputes softmax on-the-fly (same as forward) and computes gradients.
// No weight saving, no CPU readback.
//
// Layout: BHSD (batch, head, seq, dim) for Q, K, V, dO, dQ, dK, dV.
// dK and dV use atomicAdd (multiple queries contribute to same key).
//
// Push constants (24 bytes):
//   uint batch_size, num_heads, seq_len, head_dim, window_size;
//   float scale;
//
// Uses shared memory for dot-product reduction (correct on both wave32 and wave64).

#extension GL_EXT_shader_atomic_float : enable

layout(local_size_x = 64) in;

layout(set=0, binding=0) readonly buffer Queries { float Q[]; };
layout(set=0, binding=1) readonly buffer Keys { float K[]; };
layout(set=0, binding=2) readonly buffer Values { float V[]; };
layout(set=0, binding=3) readonly buffer GradOut { float dO[]; };
layout(set=0, binding=4) buffer GradQuery { float dQ[]; };
layout(set=0, binding=5) buffer GradKey { float dK[]; };
layout(set=0, binding=6) buffer GradVal { float dV[]; };

layout(push_constant) uniform Params {
    uint batch_size;
    uint num_heads;
    uint seq_len;
    uint head_dim;
    uint window_size;
    float scale;
};

shared float sm_reduce[64];

float dot_product(float a_reg[4], float b_reg[4], uint dpt, uint tid, uint head_dim) {
    // Each thread computes partial dot, then reduce via shared memory
    float partial = 0.0;
    for (uint i = 0u; i < dpt && tid + i * 64u < head_dim; i++) {
        partial += a_reg[i] * b_reg[i];
    }
    sm_reduce[tid] = partial;
    barrier();
    if (tid < 32u) sm_reduce[tid] += sm_reduce[tid + 32u];
    barrier();
    if (tid < 16u) sm_reduce[tid] += sm_reduce[tid + 16u];
    barrier();
    if (tid < 8u) sm_reduce[tid] += sm_reduce[tid + 8u];
    barrier();
    if (tid < 4u) sm_reduce[tid] += sm_reduce[tid + 4u];
    barrier();
    if (tid < 2u) sm_reduce[tid] += sm_reduce[tid + 2u];
    barrier();
    if (tid < 1u) sm_reduce[tid] += sm_reduce[tid + 1u];
    barrier();
    return sm_reduce[0];
}

void main() {
    uint wid = gl_WorkGroupID.x;
    uint tid = gl_LocalInvocationID.x;

    uint bh = wid / seq_len;
    uint q_row = wid % seq_len;
    if (q_row >= seq_len) return;

    uint q_base = bh * seq_len * head_dim + q_row * head_dim;
    uint k_end = q_row;
    uint k_start = (q_row + 1u >= window_size) ? (q_row + 1u - window_size) : 0u;
    uint kv_len = k_end - k_start + 1u;
    uint kv_base = bh * seq_len * head_dim;

    // Load Q and dO rows into registers
    float q_reg[4];
    float do_reg[4];
    uint dpt = (head_dim + 63u) / 64u;
    for (uint i = 0u; i < dpt && tid + i * 64u < head_dim; i++) {
        q_reg[i] = Q[q_base + tid + i * 64u];
        do_reg[i] = dO[q_base + tid + i * 64u];
    }

    // ── Pass 1: compute scores and softmax ────────────────────────────
    float scores[256];  // max window 256
    float d_attn[256];
    float row_max = -1e30;

    // Load K row into temp register for dot_product calls
    float k_reg[4];
    float v_reg[4];

    for (uint k = 0u; k < kv_len; k++) {
        uint k_pos = k_start + k;
        uint k_off = kv_base + k_pos * head_dim;

        // Load K row
        for (uint i = 0u; i < dpt && tid + i * 64u < head_dim; i++) {
            k_reg[i] = K[k_off + tid + i * 64u];
        }

        // Score = Q·K * scale via shared memory reduction
        float score = dot_product(q_reg, k_reg, dpt, tid, head_dim) * scale;
        scores[k] = score;
        row_max = max(row_max, score);
    }

    // Softmax
    float s_sum = 0.0;
    for (uint k = 0u; k < kv_len; k++) {
        float p = exp(scores[k] - row_max);
        scores[k] = p;
        s_sum += p;
    }
    float inv_sum = 1.0 / max(s_sum, 1e-8);
    for (uint k = 0u; k < kv_len; k++) {
        scores[k] *= inv_sum;
    }

    // ── Pass 2: compute d_attn, dV, d_scores, dQ, dK ─────────────────

    // d_attn[q,k] = sum_d dO[q,d] * V[k,d]
    for (uint k = 0u; k < kv_len; k++) {
        uint k_pos = k_start + k;
        uint k_off = kv_base + k_pos * head_dim;
        for (uint i = 0u; i < dpt && tid + i * 64u < head_dim; i++) {
            v_reg[i] = V[k_off + tid + i * 64u];
        }
        d_attn[k] = dot_product(do_reg, v_reg, dpt, tid, head_dim);
    }

    // dot_i = sum_k softmax[q,k] * d_attn[q,k]  (scalar, same for all threads)
    float dot_i = 0.0;
    for (uint k = 0u; k < kv_len; k++) {
        dot_i += scores[k] * d_attn[k];
    }

    // dV[k] += softmax[q,k] * dO[q]  (atomicAdd)
    for (uint k = 0u; k < kv_len; k++) {
        uint k_pos = k_start + k;
        uint k_off = kv_base + k_pos * head_dim;
        float w = scores[k];
        for (uint i = 0u; i < dpt && tid + i * 64u < head_dim; i++) {
            atomicAdd(dV[k_off + tid + i * 64u], w * do_reg[i]);
        }
    }

    // d_scores[q,k] = softmax[q,k] * (d_attn[q,k] - dot_i) * scale
    // dQ[q] += sum_k d_scores[q,k] * K[k]
    // dK[k] += d_scores[q,k] * Q[q]  (atomicAdd)
    float dq_reg[4] = float[4](0.0, 0.0, 0.0, 0.0);
    for (uint k = 0u; k < kv_len; k++) {
        uint k_pos = k_start + k;
        uint k_off = kv_base + k_pos * head_dim;
        float ds = scores[k] * (d_attn[k] - dot_i) * scale;
        for (uint i = 0u; i < dpt && tid + i * 64u < head_dim; i++) {
            dq_reg[i] += ds * K[k_off + tid + i * 64u];
            atomicAdd(dK[k_off + tid + i * 64u], ds * q_reg[i]);
        }
    }

    // Write dQ
    for (uint i = 0u; i < dpt && tid + i * 64u < head_dim; i++) {
        dQ[q_base + tid + i * 64u] = dq_reg[i];
    }
}
