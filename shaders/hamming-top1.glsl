#version 450
#extension GL_ARB_gpu_shader_int64           : require
#extension GL_EXT_shader_atomic_int64        : require
#extension GL_KHR_shader_subgroup_arithmetic : require

// -- Ultimate GPU argmin Hamming search ---------------------------------------
//
// Three architectural optimizations fused into one kernel:
//
// 1. LDS (Shared Memory): Load the 1.3 KB query into L1 shared memory once
//    per workgroup. All threads read query at Infinity Fabric speed (~19
//    TB/s), leaving the global memory bus 100% available for the cache.
//
// 2. 128-bit Vectorized Loads: uvec4 maps to RDNA 2's native
//    buffer_load_dwordx4. 16 bytes per instruction instead of 4.
//
// 3. atomicMin on packed uint64: (distance << 32) | index
//    GPU-side argmin returns 8 bytes instead of N×4.
//
// Subgroup-agnostic: works with Wave32 (8 entries/WG) or Wave64 (4 entries/WG).
// Uses gl_SubgroupSize and gl_SubgroupID instead of hardcoded 32.
//
// Dispatch: workgroups_x = ceil(N / entries_per_wg)
//           entries_per_wg = push.num_queries (set by CPU based on subgroup size)

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(std430, set = 0, binding = 0) readonly buffer QueryBuf {
    uvec4 query_bits[];
};
layout(std430, set = 0, binding = 1) readonly buffer CacheBuf {
    uvec4 cache_bits[];
};
layout(std430, set = 0, binding = 2) coherent buffer ResultBuf {
    uint64_t best_packed;  // upper 32 = distance, lower 32 = index
};

layout(push_constant) uniform PushConsts {
    uint num_entries;       // Total cache entries
    uint words_per_vec;     // uint32 words per vector (e.g. 320 for d=10240)
    uint entries_per_wg;    // 256 / subgroup_size (4 for Wave64, 8 for Wave32)
    uint query_offset;      // (unused, reserved)
};

// L1 LDS: supports up to 32,768-dimensional vectors (256 * 128 bits)
shared uvec4 shared_query[256];

void main() {
    // 320 uints / 4 = 80 uvec4 loads per vector
    uint vec4Count = words_per_vec / 4;

    // 1. Cooperative query load into LDS
    for (uint i = gl_LocalInvocationID.x; i < vec4Count; i += gl_WorkGroupSize.x) {
        shared_query[i] = query_bits[i];
    }
    barrier();

    // 2. Use hardware subgroup ID — correct regardless of Wave32 or Wave64
    uint entryIdx = gl_WorkGroupID.x * entries_per_wg + gl_SubgroupID;
    if (entryIdx >= num_entries) return;

    // 3. Strided uvec4 loads — subgroupSize threads read contiguous cache data
    uint offset = entryIdx * vec4Count;
    uint local_dist = 0;

    for (uint i = gl_SubgroupInvocationID; i < vec4Count; i += gl_SubgroupSize) {
        uvec4 c = cache_bits[offset + i];
        uvec4 q = shared_query[i];
        ivec4 bc = bitCount(c ^ q);
        local_dist += uint(bc.x + bc.y + bc.z + bc.w);
    }

    // 4. Hardware cross-lane reduction (1 cycle on RDNA 2)
    uint total_dist = subgroupAdd(local_dist);

    // 5. Subgroup leader writes to global atomicMin
    if (gl_SubgroupInvocationID == 0) {
        uint64_t packed = (uint64_t(total_dist) << 32) | uint64_t(entryIdx);
        atomicMin(best_packed, packed);
    }
}
