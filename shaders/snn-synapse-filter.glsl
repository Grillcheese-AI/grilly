#version 450

/*
 * SNN Synapse Filter — Exponential decay temporal smoothing
 *
 * y[t] = y[t-1] * exp(-1/tau) + x[t]
 *
 * Bindings:
 *   0: x_in    [n_elements] (read — input at current timestep)
 *   1: y_state [n_elements] (read/write — filter state)
 *
 * Push constants:
 *   n_elements (uint)
 *   decay      (float) = exp(-1/tau)
 */

layout(local_size_x = 256) in;

layout(set = 0, binding = 0) readonly buffer InputBuf { float x_in[]; };
layout(set = 0, binding = 1) buffer StateBuf          { float y_state[]; };

layout(push_constant) uniform PushConstants {
    uint  n_elements;
    float decay;
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= n_elements) return;

    y_state[idx] = y_state[idx] * decay + x_in[idx];
}
