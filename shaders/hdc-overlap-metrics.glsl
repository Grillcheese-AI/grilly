#version 450
#extension GL_KHR_shader_subgroup_arithmetic : require

// Overlap and Jaccard similarity for packed binary hypervectors.
//
// Given a query and a codebook of N entries, computes per-entry:
//   overlap   = |A & B|                (raw intersection count)
//   jaccard   = |A & B| / |A | B|     (Jaccard index)
//   overlap_c = |A & B| / min(|A|,|B|) (overlap coefficient)
//
// Uses bitCount (single-cycle popcount on RDNA 2) + subgroup reduction.
//
// Dispatch: workgroups_x = num_entries
//           local_size_x = threads per entry (should divide words_per_vec)
//
// Output layout: results[entry * 3 + 0] = overlap
//                results[entry * 3 + 1] = jaccard
//                results[entry * 3 + 2] = overlap_coefficient

layout(local_size_x = 64) in;

layout(std430, set = 0, binding = 0) readonly buffer Query    { uint query[]; };
layout(std430, set = 0, binding = 1) readonly buffer Codebook { uint codebook[]; };
layout(std430, set = 0, binding = 2) writeonly buffer Results { float results[]; };

layout(push_constant) uniform Params {
    uint words_per_vec;   // dim / 32
    uint num_entries;     // codebook size
    uint dim;             // original dimension
};

void main() {
    uint entry_idx = gl_WorkGroupID.x;
    if (entry_idx >= num_entries) return;

    uint tid = gl_LocalInvocationID.x;
    uint base = entry_idx * words_per_vec;

    uint local_and = 0u;   // |A & B|
    uint local_or  = 0u;   // |A | B|
    uint local_a   = 0u;   // |A| (query popcount)
    uint local_b   = 0u;   // |B| (entry popcount)

    // Strided accumulation across words
    for (uint w = tid; w < words_per_vec; w += gl_WorkGroupSize.x) {
        uint a = query[w];
        uint b = codebook[base + w];
        local_and += bitCount(a & b);
        local_or  += bitCount(a | b);
        local_a   += bitCount(a);
        local_b   += bitCount(b);
    }

    // Cross-lane reduction
    uint total_and = subgroupAdd(local_and);
    uint total_or  = subgroupAdd(local_or);
    uint total_a   = subgroupAdd(local_a);
    uint total_b   = subgroupAdd(local_b);

    if (subgroupElect()) {
        float f_and = float(total_and);
        float f_or  = float(total_or);
        float f_a   = float(total_a);
        float f_b   = float(total_b);

        // Overlap (raw count, normalized by dim)
        results[entry_idx * 3u + 0u] = f_and / float(dim);

        // Jaccard: |A&B| / |A|B|  (0 if both empty)
        results[entry_idx * 3u + 1u] = (f_or > 0.0) ? (f_and / f_or) : 0.0;

        // Overlap coefficient: |A&B| / min(|A|, |B|)  (0 if either empty)
        float min_ab = min(f_a, f_b);
        results[entry_idx * 3u + 2u] = (min_ab > 0.0) ? (f_and / min_ab) : 0.0;
    }
}
