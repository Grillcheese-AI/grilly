#version 450
// spike-scatter.glsl — event-driven sparse synaptic propagation (GATHER form).
// One thread per POST neuron; each loops the fired-pre list and accumulates.
// No atomics (each I_acc[post] written by exactly one thread); adjacent threads
// read consecutive W[pre*N+post] -> coalesced. Cost = fired_count * N (work-
// proportional to spike activity). Grid = ceil(N / 256).

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer FiredIdx   { uint  fired_idx[]; };
layout(set = 0, binding = 1) readonly buffer FiredCount { uint  fired_count[]; };
layout(set = 0, binding = 2) readonly buffer Weights    { float W[]; };      // [pre*N + post]
layout(set = 0, binding = 3)          buffer InputAccum { float I_acc[]; };

layout(push_constant) uniform PushConsts { uint N; };

void main() {
    uint post = gl_GlobalInvocationID.x;
    if (post >= N) return;
    uint nf = fired_count[0];
    float acc = 0.0;
    for (uint i = 0u; i < nf; ++i) {
        uint pre = fired_idx[i];
        acc += W[pre * N + post];
    }
    I_acc[post] = acc;
}
