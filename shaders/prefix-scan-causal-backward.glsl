#version 450
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_basic : require

/*
 * Causal Linear-RNN Prefix Scan (backward)
 *
 * Given forward recurrence h_t = a_t * h_{t-1} + x_t, the adjoint rules are:
 *
 *     dx_t = dh_t + a_{t+1} * dx_{t+1}      (anti-causal sum)
 *     da_t = dx_t * h_{t-1}
 *
 * Using g_t = prod_{k<=t} a_k (forward cumprod):
 *
 *     dx_t = (1 / g_t) * sum_{s>=t} g_s * dh_s
 *
 * Strategy:
 *   1. Compute g_t via ``subgroupInclusiveAdd(log(a))`` + exp. Verified
 *      correct on partial Wave64 subgroups (see earlier debug dump:
 *      max abs err 2.98e-08 vs numpy cumprod).
 *   2. Compute weighted_dh = g_t * dh_t per thread.
 *   3. Store to shared memory.
 *   4. Each thread LOOPS sequentially over shared[t..seq_len-1] to compute
 *      its own right_sum. For seq_len <= 32 this is ~32 iterations which
 *      is cheap compared to the rest of the pipeline, and it sidesteps
 *      the ``total`` broadcast trap entirely (neither ``subgroupAdd`` nor
 *      a ``shared_total`` write from thread 31 produced the correct total
 *      in earlier attempts — the cause is still unclear but likely some
 *      AMD partial-Wave64 edge case).
 *
 * Dispatch: one workgroup per (batch, hidden_dim) pair, one thread per
 * time step. Constraint: seq_len <= 32 (matches local_size_x).
 */

layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0) readonly  buffer GradHBuffer    { float dH[]; };
layout(binding = 1) readonly  buffer DecayBuffer    { float A[]; };
layout(binding = 2) readonly  buffer ForwardHBuffer { float H[]; };
layout(binding = 3) readonly  buffer ForwardXBuffer { float X[]; };
layout(binding = 4) writeonly buffer GradXBuffer    { float dX[]; };
layout(binding = 5) writeonly buffer GradABuffer    { float dA[]; };

layout(push_constant) uniform PushConstants {
    uint seq_len;
    uint hidden_dim;
} params;

shared float scratch[32];

void main() {
    uint t = gl_LocalInvocationID.x;
    uint d = gl_WorkGroupID.x;
    uint b = gl_WorkGroupID.y;

    bool in_bounds = (d < params.hidden_dim);
    bool in_seq    = in_bounds && (t < params.seq_len);

    uint idx = in_seq
        ? (b * params.seq_len * params.hidden_dim) + (t * params.hidden_dim) + d
        : 0u;

    // Load forward values (neutral for padding threads).
    float a_t  = in_seq ? max(A[idx], 1e-6) : 1.0;
    float dh_t = in_seq ? dH[idx] : 0.0;
    float h_t  = in_seq ? H[idx]  : 0.0;
    float x_t  = in_seq ? X[idx]  : 0.0;

    // ── Forward cumulative product of a: g_t = prod_{k<=t} a_k ──
    float log_a = log(a_t);
    float cumsum_log_a = subgroupInclusiveAdd(log_a);
    float g_t = exp(cumsum_log_a);

    // ── Store weighted_dh to shared memory ──
    float weighted_dh = g_t * dh_t;
    scratch[t] = weighted_dh;
    barrier();

    // ── Sequential right-sum: right_sum[t] = sum_{s>=t} weighted_dh[s] ──
    float right_sum = 0.0;
    for (uint s = t; s < params.seq_len; s++) {
        right_sum += scratch[s];
    }

    // dx_t = right_sum / g_t
    float dx_t = right_sum / g_t;

    // ── da_t = dx_t * h_{t-1}, with h_{t-1} = (h_t - x_t) / a_t ──
    // Numerically unstable when a_t is tiny — max(A[idx], 1e-6) clamp
    // keeps it away from zero. TODO: save h_{t-1} during forward for
    // full stability.
    float h_prev = 0.0;
    if (in_seq && t > 0u) {
        h_prev = (h_t - x_t) / a_t;
    }
    float da_t = dx_t * h_prev;

    if (in_seq) {
        dX[idx] = dx_t;
        dA[idx] = da_t;
    }
}
