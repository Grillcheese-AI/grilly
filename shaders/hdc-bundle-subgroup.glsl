#version 450
#extension GL_KHR_shader_subgroup_ballot : require

// Subgroup ballot-based majority bundling for packed binary hypervectors.
//
// Replaces the per-bit loop in hdc-bundle-packed.glsl with hardware
// ballot instructions. Each subgroup lane owns one input vector;
// subgroupBallot counts votes across lanes in a single instruction.
//
// Dispatch: workgroups_x = ceil(words_per_vec / local_size_x)
//           Each workgroup processes a slice of the output words.
//
// Constraint: num_vectors <= gl_SubgroupSize (32 or 64 on RDNA 2).
// For num_vectors > SubgroupSize, use hdc-bundle-packed.glsl instead.

layout(local_size_x = 256) in;

layout(std430, set = 0, binding = 0) readonly buffer Vectors { uint vectors[]; };
layout(std430, set = 0, binding = 1) writeonly buffer Out    { uint out_data[]; };

layout(push_constant) uniform Params {
    uint words_per_vec;   // dim / 32
    uint num_vectors;     // N (must be <= SubgroupSize)
};

void main() {
    uint word_idx = gl_GlobalInvocationID.x;
    if (word_idx >= words_per_vec) return;

    uint threshold = num_vectors / 2u;
    uint result = 0u;

    // Each lane loads its vector's word (lane >= num_vectors loads 0)
    uint my_word = 0u;
    uint lane = gl_SubgroupInvocationID;
    if (lane < num_vectors) {
        my_word = vectors[lane * words_per_vec + word_idx];
    }

    // For each bit position, ballot across lanes to count votes
    for (uint bit = 0u; bit < 32u; bit++) {
        uint mask = 1u << bit;
        bool vote = (my_word & mask) != 0u;

        // subgroupBallot returns a uvec4 bitmask of which lanes voted true
        uvec4 ballot = subgroupBallot(vote);

        // Count set bits in the ballot (only the first num_vectors lanes matter)
        uint count = bitCount(ballot.x);
        if (gl_SubgroupSize > 32u) {
            count += bitCount(ballot.y);
        }

        if (count > threshold) {
            result |= mask;
        }
    }

    // Only first lane writes (all lanes computed the same result)
    if (gl_SubgroupInvocationID == 0u) {
        out_data[word_idx] = result;
    }
}
