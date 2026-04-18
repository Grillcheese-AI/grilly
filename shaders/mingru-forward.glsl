#version 450

/*
 * MinGRU Fused Activation + Causal Scan (forward)
 *
 * Fuses the following computations:
 *   x_scan = sigmoid(g) * tanh(v)
 *   a      = 0.001 + 0.998 * sigmoid(d)
 *   h_t    = a_t * h_{t-1} + x_scan_t
 *
 * Inputs: G, V, D (B, S, D)
 * Output: H (B, S, D)
 *
 * Strategy: one thread per (batch, hidden_dim) pair, sequential time loop.
 */

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0) readonly  buffer GateG  { float G[]; };
layout(binding = 1) readonly  buffer GateV  { float V[]; };
layout(binding = 2) readonly  buffer GateD  { float D[]; };
layout(binding = 3) writeonly buffer Output { float H[]; };

layout(push_constant) uniform PushConstants {
    uint seq_len;
    uint hidden_dim;
} params;

float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

void main() {
    uint d = gl_GlobalInvocationID.x;  // hidden dim index
    uint b = gl_WorkGroupID.y;         // batch index

    if (d >= params.hidden_dim) return;

    uint stride_t = params.hidden_dim;
    uint base     = b * params.seq_len * params.hidden_dim + d;

    float h = 0.0;
    for (uint t = 0u; t < params.seq_len; ++t) {
        uint idx = base + t * stride_t;
        
        float g_t = G[idx];
        float v_t = V[idx];
        float d_t = D[idx];
        
        float x_scan_t = sigmoid(g_t) * tanh(v_t);
        float a_t      = 0.001 + 0.998 * sigmoid(d_t);
        
        h = a_t * h + x_scan_t;
        H[idx] = h;
    }
}
