#version 450

// ── Scalar Hamming Distance Search ───────────────────────────────────
//
// One thread per cache entry. No subgroup operations.
//
// Rationale: The subgroup approach (1 entry per wave) gives only 2
// coalesced cache line requests per iteration — too few to hide GDDR6
// memory latency (~100ns). With 640 concurrent waves × 2 cache lines,
// the GPU has ~1,280 outstanding requests, but needs ~300K+ to
// saturate 432 GB/s.
//
// This scalar approach: 64 threads per wave, each processing a
// DIFFERENT entry. In each iteration, the wave issues 64 INDEPENDENT
// cache line requests (one per thread, 1280 bytes apart). This is 32x
// more outstanding requests than the subgroup approach.
//
// Tradeoff: Non-coalesced access (scattered cache lines). But the
// massive increase in memory-level parallelism should more than
// compensate on GDDR6 with its 12 memory channels and deep queues.
//
// Dispatch: workgroups_x = ceil(N / 256)

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer QueryBuf  { uint query_packed[];  };
layout(set = 0, binding = 1) readonly buffer CacheBuf  { uint cache_packed[];  };
layout(set = 0, binding = 2) writeonly buffer DistBuf  { uint distances[];     };

layout(push_constant) uniform PushConsts {
    uint num_entries;
    uint words_per_vec;
    uint num_queries;
    uint query_offset;
};

void main() {
    uint entry = gl_GlobalInvocationID.x;
    if (entry >= num_entries) return;

    uint offset = entry * words_per_vec;
    uint qb     = query_offset * words_per_vec;
    uint dist   = 0;

    // Sequential loop: each thread processes its entire entry
    // The compiler can pipeline multiple iterations' loads
    for (uint w = 0; w < words_per_vec; w++) {
        dist += bitCount(query_packed[qb + w] ^ cache_packed[offset + w]);
    }

    distances[entry] = dist;
}
