#version 450

/*
 * Causal Linear-RNN Prefix Scan (backward)
 *
 * Given forward recurrence h_t = a_t * h_{t-1} + x_t, adjoint rules are:
 *
 *     dx_t = dh_t + a_{t+1} * dx_{t+1}   (anti-causal recurrence; dx_T+1 = 0)
 *     da_t = dx_t * h_{t-1}              (h_{-1} = 0)
 *
 * Strategy (rev 2, 2026-04-15): one thread per (batch, hidden_dim) pair,
 * sequential reverse-time loop. Matches the new per-thread forward shader
 * and drops the subgroup-scan seq-length cap.
 *
 *   - h_{t-1} comes from the saved forward H buffer (no reconstruction).
 *   - a_{t+1} is read fresh each iteration; for t = seq_len-1 the
 *     incoming dx_next is 0 so the factor does not matter.
 *   - The same ``max(A[idx], 1e-6)`` clamp as forward keeps gradients
 *     bounded when a decay gate sits at the edge of its range.
 *
 * Dispatch matches forward: local_size_x=64 tile of hidden_dim,
 * workgroup grid (ceil(hidden_dim/64), batch).
 */

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

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

void main() {
    uint d = gl_GlobalInvocationID.x;
    uint b = gl_WorkGroupID.y;

    if (d >= params.hidden_dim) return;

    uint stride_t = params.hidden_dim;
    uint base     = b * params.seq_len * params.hidden_dim + d;

    float dx_next = 0.0;
    // Iterate t from seq_len-1 down to 0. Use a signed counter so the
    // termination check (t >= 0) is safe for the unsigned loop.
    for (int t_signed = int(params.seq_len) - 1; t_signed >= 0; --t_signed) {
        uint t   = uint(t_signed);
        uint idx = base + t * stride_t;

        float dh_t = dH[idx];

        // a_{t+1} factor between dx_{t+1} and its contribution to dx_t.
        float a_tp1 = 1.0;  // irrelevant when dx_next == 0 (t == seq_len-1)
        if (t + 1u < params.seq_len) {
            uint idx_tp1 = base + (t + 1u) * stride_t;
            a_tp1 = max(A[idx_tp1], 1e-6);
        }
        float dx_t = dh_t + a_tp1 * dx_next;
        dX[idx] = dx_t;

        // h_{t-1}: zero at t == 0, otherwise read the forward hidden state.
        float h_prev = 0.0;
        if (t > 0u) {
            uint idx_prev = base + (t - 1u) * stride_t;
            h_prev = H[idx_prev];
        }
        dA[idx] = dx_t * h_prev;

        dx_next = dx_t;
    }
}
