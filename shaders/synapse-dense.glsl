#version 450
// synapse-dense.glsl — dense synaptic propagation, matched to spike-scatter's
// gather structure: one thread per POST, loop ALL pre. Same coalesced W access;
// only the loop length differs (N here vs fired_count in the scatter). This makes
// the A/B a clean function of activity.
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer Spikes     { float spikes[]; };
layout(set = 0, binding = 1) readonly buffer Weights    { float W[]; };   // [pre*N + post]
layout(set = 0, binding = 2)          buffer InputAccum { float I_acc[]; };

layout(push_constant) uniform PushConsts { uint N; };

void main() {
    uint post = gl_GlobalInvocationID.x;
    if (post >= N) return;
    float acc = 0.0;
    for (uint pre = 0u; pre < N; ++pre) {
        acc += spikes[pre] * W[pre * N + post];
    }
    I_acc[post] = acc;
}
