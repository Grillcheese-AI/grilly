#version 450
#extension GL_KHR_shader_subgroup_arithmetic : require

// Top-K Hamming search for packed binary hypervectors.
//
// Returns the K nearest entries (smallest Hamming distance) from a cache
// of N packed binary vectors. Each workgroup processes one cache entry,
// computes its Hamming distance via bitCount + subgroup reduction, then
// writes (distance, index) to a results buffer for CPU-side top-k selection.
//
// Unlike hamming-top1.glsl which uses atomicMin for a single winner,
// this shader writes all distances so the host can extract top-K.
// For large N, a GPU-side bitonic sort pass can be added.
//
// Dispatch: workgroups_x = ceil(num_entries / entries_per_wg)
//           entries_per_wg = local_size_x / subgroup_size

layout(local_size_x = 256) in;

layout(std430, set = 0, binding = 0) readonly buffer QueryBuf { uint query[]; };
layout(std430, set = 0, binding = 1) readonly buffer CacheBuf { uint cache[]; };
layout(std430, set = 0, binding = 2) writeonly buffer DistBuf { uint distances[]; };

layout(push_constant) uniform Params {
    uint words_per_vec;     // dim / 32
    uint num_entries;       // total cache entries
    uint entries_per_wg;    // local_size_x / subgroup_size
};

// LDS query cache (supports up to 32768-dim vectors = 1024 words)
shared uint shared_query[1024];

void main() {
    // 1. Cooperative query load into LDS
    for (uint i = gl_LocalInvocationID.x; i < words_per_vec; i += gl_WorkGroupSize.x) {
        shared_query[i] = query[i];
    }
    barrier();

    // 2. Each subgroup handles one cache entry
    uint entry_idx = gl_WorkGroupID.x * entries_per_wg + gl_SubgroupID;
    if (entry_idx >= num_entries) return;

    // 3. Strided Hamming distance computation
    uint base = entry_idx * words_per_vec;
    uint local_dist = 0u;

    for (uint w = gl_SubgroupInvocationID; w < words_per_vec; w += gl_SubgroupSize) {
        local_dist += bitCount(cache[base + w] ^ shared_query[w]);
    }

    // 4. Cross-lane reduction
    uint total_dist = subgroupAdd(local_dist);

    // 5. Subgroup leader writes distance (host does top-k sort)
    if (gl_SubgroupInvocationID == 0u) {
        distances[entry_idx] = total_dist;
    }
}
