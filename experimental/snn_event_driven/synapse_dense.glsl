#version 450
// Dense synaptic propagation baseline: I_acc[post] = sum_pre spikes[pre]*W[pre,post].
// One thread per post neuron; reads all N*N weights every step (O(N^2)).
// A/B partner for synapse_scatter.glsl.
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer Spikes     { float spikes[]; };
layout(set = 0, binding = 1) readonly buffer Weights    { float W[]; };       // [pre*N + post]
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
