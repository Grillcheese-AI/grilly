#version 450
// Sliding-window causal attention (0.0.2)
//
// Single-pass, one workgroup per (batch, head, q_pos).
// Iterates K over the causal window [max(0, q-W+1), q].
// Uses online softmax — no O(S²) materialization.
//
// Layout: BHSD (batch, head, seq, dim) for Q, K, V, O.
//
// Push constants (24 bytes):
//   uint batch_size, num_heads, seq_len, head_dim, window_size;
//   float scale;   // 1/sqrt(head_dim)
//
// Uses shared memory for dot-product reduction (correct on both wave32 and wave64).

layout(local_size_x = 64) in;

layout(set=0, binding=0) readonly buffer Queries { float Q[]; };
layout(set=0, binding=1) readonly buffer Keys { float K[]; };
layout(set=0, binding=2) readonly buffer Values { float V[]; };
layout(set=0, binding=3) writeonly buffer Output { float O[]; };

layout(push_constant) uniform Params {
    uint batch_size;
    uint num_heads;
    uint seq_len;
    uint head_dim;
    uint window_size;
    float scale;
};

shared float sm_reduce[64];

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

    // Load Q row — each thread handles head_dim/64 elements (1 when Dh=64)
    float q_reg[4];
    uint dpt = (head_dim + 63u) / 64u;
    for (uint i = 0u; i < dpt && tid + i * 64u < head_dim; i++) {
        q_reg[i] = Q[q_base + tid + i * 64u];
    }

    // Online softmax
    float row_max = -1e30;
    float row_sum = 0.0;
    float o_reg[4] = float[4](0.0, 0.0, 0.0, 0.0);

    for (uint k = 0u; k < kv_len; k++) {
        uint k_pos = k_start + k;
        uint k_off = kv_base + k_pos * head_dim;

        // Dot product Q·K via shared memory reduction (works on wave32 and wave64)
        float partial = 0.0;
        for (uint i = 0u; i < dpt && tid + i * 64u < head_dim; i++) {
            partial += q_reg[i] * K[k_off + tid + i * 64u];
        }
        sm_reduce[tid] = partial;
        barrier();

        // Tree reduction in shared memory
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

        float score = sm_reduce[0] * scale;

        // Online softmax update
        float old_max = row_max;
        row_max = max(row_max, score);
        float rescale = exp(old_max - row_max);
        row_sum *= rescale;
        for (uint i = 0u; i < dpt; i++) {
            o_reg[i] *= rescale;
        }
        float p = exp(score - row_max);
        row_sum += p;

        // Accumulate weighted V
        for (uint i = 0u; i < dpt && tid + i * 64u < head_dim; i++) {
            o_reg[i] += p * V[k_off + tid + i * 64u];
        }
        barrier();  // ensure all threads done reading sm_reduce before next k
    }

    // Finalize: divide by softmax sum
    float inv_sum = 1.0 / max(row_sum, 1e-8);
    for (uint i = 0u; i < dpt && tid + i * 64u < head_dim; i++) {
        O[q_base + tid + i * 64u] = o_reg[i] * inv_sum;
    }
}
