#version 450
// Sum of squares reduction: acc[0] += sum_i x[i]^2  (atomic accumulate).
// One workgroup reduces 256 elements in shared memory, then a single float
// atomicAdd to the global accumulator. Call once per buffer with the SAME acc to
// get a global sum across many buffers (e.g. a global gradient L2 norm) with a
// single 4-byte readback instead of pulling every grad buffer back to host.
#extension GL_EXT_shader_atomic_float : require

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer In  { float x[]; };
layout(set = 0, binding = 1)          buffer Acc { float acc[]; };  // acc[0]

layout(push_constant) uniform PushConsts { uint n; };

shared float partial[256];

void main() {
    uint gid = gl_GlobalInvocationID.x;
    uint lid = gl_LocalInvocationID.x;

    float v = 0.0;
    if (gid < n) { float g = x[gid]; v = g * g; }
    partial[lid] = v;
    barrier();

    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (lid < s) partial[lid] += partial[lid + s];
        barrier();
    }
    if (lid == 0u) atomicAdd(acc[0], partial[0]);
}
