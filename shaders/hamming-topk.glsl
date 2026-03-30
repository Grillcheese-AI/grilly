#version 450
#extension GL_KHR_shader_subgroup_arithmetic : require

// Optimized Top-K Hamming search for packed binary hypervectors.
//
// RDNA 2 optimizations:
//   1. uvec4 vectorized loads: 128-bit fetches (4× fewer transactions)
//   2. Dynamic subgroup size: uses gl_SubgroupSize (safe for Wave32/Wave64)
//   3. Cooperative query load: no loop when words_per_vec/4 <= workgroup_size
//   4. Bank-conflict-free LDS access via uvec4 alignment
//
// Dispatch: workgroups_x = ceil(num_entries / entries_per_wg)

layout(local_size_x = 256) in;

// 128-bit vectorized buffers
layout(std430, set = 0, binding = 0) readonly buffer QueryBuf { uvec4 query[]; };
layout(std430, set = 0, binding = 1) readonly buffer CacheBuf { uvec4 cache[]; };
layout(std430, set = 0, binding = 2) writeonly buffer DistBuf { uint distances[]; };

layout(push_constant) uniform Params {
    uint vec4s_per_vec;     // words_per_vec / 4 (e.g., 320/4 = 80 for d=10240)
    uint num_entries;       // total cache entries
};

// LDS query cache — uvec4 aligned (supports up to 32768-dim vectors = 256 uvec4s)
shared uvec4 shared_query[256];

void main() {
    // 1. Cooperative query load into LDS — one uvec4 per thread
    if (gl_LocalInvocationID.x < vec4s_per_vec) {
        shared_query[gl_LocalInvocationID.x] = query[gl_LocalInvocationID.x];
    }
    barrier();

    // 2. Dynamic entries_per_wg from hardware subgroup size
    uint entries_per_wg = gl_WorkGroupSize.x / gl_SubgroupSize;
    uint entry_idx = gl_WorkGroupID.x * entries_per_wg + gl_SubgroupID;
    if (entry_idx >= num_entries) return;

    // 3. Strided Hamming distance — uvec4 = 128 bits per iteration
    uint base = entry_idx * vec4s_per_vec;
    uint local_dist = 0u;

    for (uint w = gl_SubgroupInvocationID; w < vec4s_per_vec; w += gl_SubgroupSize) {
        uvec4 c = cache[base + w];
        uvec4 q = shared_query[w];

        // 4 × bitCount on XOR'd 32-bit words = 128 bits processed per iteration
        local_dist += bitCount(c.x ^ q.x)
                    + bitCount(c.y ^ q.y)
                    + bitCount(c.z ^ q.z)
                    + bitCount(c.w ^ q.w);
    }

    // 4. Cross-lane reduction (1 cycle on RDNA 2)
    uint total_dist = subgroupAdd(local_dist);

    // 5. Subgroup leader writes distance
    if (gl_SubgroupInvocationID == 0u) {
        distances[entry_idx] = total_dist;
    }
}
