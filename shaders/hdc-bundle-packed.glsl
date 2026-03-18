#version 450
layout(local_size_x = 256) in;
// N vectors packed as uint32, laid out: vec0_word0, vec0_word1, ..., vec1_word0, vec1_word1, ...
layout(set=0, binding=0) readonly buffer Vectors { uint vectors[]; };
layout(set=0, binding=1) writeonly buffer Out { uint out_data[]; };
layout(push_constant) uniform Params {
    uint words_per_vec;  // dim / 32
    uint num_vectors;    // N
};

void main() {
    uint word_idx = gl_GlobalInvocationID.x;
    if (word_idx >= words_per_vec) return;

    uint result = 0u;
    uint threshold = num_vectors / 2u;

    // For each bit position in this word
    for (uint bit = 0u; bit < 32u; bit++) {
        uint mask = 1u << bit;
        uint count = 0u;
        // Count votes across all vectors
        for (uint v = 0u; v < num_vectors; v++) {
            if ((vectors[v * words_per_vec + word_idx] & mask) != 0u) {
                count++;
            }
        }
        if (count > threshold) {
            result |= mask;
        }
    }
    out_data[word_idx] = result;
}
