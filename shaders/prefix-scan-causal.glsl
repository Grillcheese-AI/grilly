#version 450

/*
 * Causal Linear-RNN Prefix Scan (forward)
 *
 * Computes strictly-causal recurrence:
 *     h_t = a_t * h_{t-1} + x_t
 * where a_t is a per-(batch, seq, dim) decay gate in (0, 1], and h_0 = 0.
 *
 * Strategy (rev 2, 2026-04-15): one thread per (batch, hidden_dim) pair,
 * sequential time loop. Drops the previous subgroupInclusiveAdd scan path
 * that capped sequence length at the subgroup size (32 Wave32 / 64 Wave64).
 * The new shader works for any seq_len; parallelism comes from the
 * (batch × hidden_dim) grid instead of within a subgroup.
 *
 * Dispatch:
 *   local_size_x = 64  (tile of hidden_dim per workgroup, good AMD wave32/64 fit)
 *   gl_WorkGroupID.x = hidden_dim tile index
 *   gl_WorkGroupID.y = batch index
 *   gl_LocalInvocationID.x = within-tile hidden_dim offset (0..63)
 *
 * Buffer layout is (B, S, D) row-major:
 *     index(b, t, d) = b * S * D + t * D + d
 */

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0) readonly  buffer InputBuffer  { float X[]; };
layout(binding = 1) readonly  buffer DecayBuffer  { float A[]; };
layout(binding = 2) writeonly buffer OutputBuffer { float H[]; };

layout(push_constant) uniform PushConstants {
    uint seq_len;
    uint hidden_dim;
} params;

void main() {
    uint d = gl_GlobalInvocationID.x;  // hidden dim index
    uint b = gl_WorkGroupID.y;         // batch index

    if (d >= params.hidden_dim) return;

    uint stride_t = params.hidden_dim;
    uint base     = b * params.seq_len * params.hidden_dim + d;

    float h = 0.0;
    for (uint t = 0u; t < params.seq_len; ++t) {
        uint idx = base + t * stride_t;
        float a_t = max(A[idx], 1e-6);
        float x_t = X[idx];
        h = a_t * h + x_t;
        H[idx] = h;
    }
}
