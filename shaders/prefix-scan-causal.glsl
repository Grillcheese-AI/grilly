#version 450
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_basic : require

/*
 * Causal Linear-RNN Prefix Scan (forward)
 *
 * Computes strictly-causal recurrence in parallel:
 *     h_t = a_t * h_{t-1} + x_t
 * where a_t is a per-(batch, seq, dim) decay gate in (0, 1], and h_0 = 0.
 *
 * Math trick: taking the cumulative product of a_t and scaling x_t by its
 * inverse turns the recurrence into two commutative scans — a cumulative
 * sum of log(a) for the decay, and a cumulative sum of the scaled input.
 *
 *     cumprod_a[t] = prod_{s<=t} a_s = exp(sum_{s<=t} log(a_s))
 *     h_t          = cumprod_a[t] * sum_{s<=t} (x_s / cumprod_a[s])
 *
 * Both scans are hardware-accelerated by ``subgroupInclusiveAdd``, which
 * runs entirely in the subgroup's registers (no shared/global traffic).
 *
 * Dispatch layout (one subgroup per (batch, hidden_dim) pair):
 *   local_size_x = subgroup_size (32 on RDNA Wave32, or use 64 for Wave64)
 *   gl_WorkGroupID.x = hidden_dim index
 *   gl_WorkGroupID.y = batch index
 *   local_x          = time step t
 *
 * Buffer layout is (B, S, D) row-major:
 *     index(b, t, d) = b * S * D + t * D + d
 *
 * Constraint: sequence length S must be <= subgroup_size. For longer
 * sequences the caller should split the sequence into chunks that fit
 * in one subgroup, using the tail of each chunk as the carry for the
 * next. A multi-subgroup version of this shader (hierarchical scan) is
 * a follow-up.
 */

layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0) readonly buffer InputBuffer  { float X[]; };
layout(binding = 1) readonly buffer DecayBuffer  { float A[]; };
layout(binding = 2) writeonly buffer OutputBuffer { float H[]; };

layout(push_constant) uniform PushConstants {
    uint seq_len;
    uint hidden_dim;
} params;

void main() {
    uint t = gl_LocalInvocationID.x;   // time step 0..seq_len-1
    uint d = gl_WorkGroupID.x;         // hidden dimension index
    uint b = gl_WorkGroupID.y;         // batch index

    if (d >= params.hidden_dim) return;
    bool in_seq = (t < params.seq_len);

    // Flattened (B, S, D) → linear index. Threads beyond seq_len write to
    // dummy index 0; we'll guard the store with ``in_seq``.
    uint idx = in_seq
        ? (b * params.seq_len * params.hidden_dim) + (t * params.hidden_dim) + d
        : 0u;

    // Load current time-step values. For padding threads (t >= seq_len),
    // fall back to neutral values: a = 1.0 (no-op in cumprod), x = 0.
    float a_t = in_seq ? max(A[idx], 1e-6) : 1.0;
    float x_t = in_seq ? X[idx] : 0.0;

    // 1. Subgroup inclusive scan of log(a) → cumulative product of a_t
    float log_a        = log(a_t);
    float cumsum_log_a = subgroupInclusiveAdd(log_a);
    float cumprod_a    = exp(cumsum_log_a);

    // 2. Scale x_t by the inverse cumprod so the next scan is a pure sum
    float scaled_x = x_t / cumprod_a;

    // 3. Subgroup inclusive scan of scaled_x
    float cumsum_scaled_x = subgroupInclusiveAdd(scaled_x);

    // 4. Undo the scaling to get h_t
    float h_t = cumprod_a * cumsum_scaled_x;

    if (in_seq) {
        H[idx] = h_t;
    }
}
