#version 450
// Event-driven sparse synaptic propagation (Tier-0 over-dispatch + early-out).
// Replaces dense  I = spikes . W  with a scatter over the fired-neuron list.
// Cost (weight reads + atomics) scales with spike count, not N^2.
// Dispatched at worst-case grid (ceil(N*N/256)); empty invocations early-out
// before touching W, so only workgroup scheduling is worst-case.
#extension GL_EXT_shader_atomic_float : require

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// binding order MUST match the `buffers` vector passed to VulkanBackend::dispatch()
layout(set = 0, binding = 0) readonly buffer FiredIdx   { uint  fired_idx[]; };
layout(set = 0, binding = 1) readonly buffer FiredCount { uint  fired_count[]; };
layout(set = 0, binding = 2) readonly buffer Weights    { float W[]; };      // [pre * N + post]
layout(set = 0, binding = 3)          buffer InputAccum { float I_acc[]; };  // zeroed each timestep

layout(push_constant) uniform PushConsts {
    uint N;   // neuron count (post dimension)
};

void main() {
    uint t  = gl_GlobalInvocationID.x;
    uint nf = fired_count[0];          // L2-resident; uniform across the workgroup
    if (t >= nf * N) return;           // empty groups return before reading W
    uint pre  = fired_idx[t / N];
    uint post = t % N;
    atomicAdd(I_acc[post], W[pre * N + post]);
}
