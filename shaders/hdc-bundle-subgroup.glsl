#version 450
#extension GL_KHR_shader_subgroup_ballot : require

// Fixed HDC Subgroup Bundle — 1 subgroup per word (was: 1 thread per word = corrupt)
//
// Bug fix: all lanes in a subgroup must process the SAME word_idx.
// Old code mapped word_idx = gl_GlobalInvocationID.x, causing each lane
// to ballot on a different word → diagonal corruption.
//
// Optimization: branchless ballot reduction, loop unroll hint.

layout(local_size_x = 256) in;

layout(std430, set = 0, binding = 0) readonly buffer Vectors { uint vectors[]; };
layout(std430, set = 0, binding = 1) writeonly buffer Out { uint out_data[]; };

layout(push_constant) uniform Params {
    uint words_per_vec;
    uint num_vectors;
};

void main() {
    // Map 1 subgroup → 1 word (uniform word_idx across all lanes)
    uint subgroups_per_wg = gl_WorkGroupSize.x / gl_SubgroupSize;
    uint global_subgroup_id = gl_WorkGroupID.x * subgroups_per_wg + gl_SubgroupID;

    uint word_idx = global_subgroup_id;
    if (word_idx >= words_per_vec) return;

    uint threshold = num_vectors / 2u;
    uint result = 0u;
    uint lane = gl_SubgroupInvocationID;

    // Each lane loads one vector's word (lane = vector index)
    uint my_word = 0u;
    if (lane < num_vectors) {
        my_word = vectors[lane * words_per_vec + word_idx];
    }

    // Ballot per bit — branchless reduction
    for (uint bit = 0u; bit < 32u; bit++) {
        uint mask = 1u << bit;
        bool vote = (my_word & mask) != 0u;
        uvec4 ballot = subgroupBallot(vote);

        // Branchless: always sum both halves (inactive lanes contribute 0)
        uint count = bitCount(ballot.x) + bitCount(ballot.y);
        if (count > threshold) {
            result |= mask;
        }
    }

    // Only lane 0 writes the cooperatively computed word
    if (lane == 0u) {
        out_data[word_idx] = result;
    }
}
