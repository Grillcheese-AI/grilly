#version 450
layout(local_size_x = 256) in;
layout(set=0, binding=0) readonly buffer Input { uint data_in[]; };
layout(set=0, binding=1) writeonly buffer Output { uint data_out[]; };
layout(push_constant) uniform Params {
    uint words_per_vec;  // dim / 32
    uint shift;          // number of bit positions to shift
};

void main() {
    uint word_idx = gl_GlobalInvocationID.x;
    if (word_idx >= words_per_vec) return;

    uint total_bits = words_per_vec * 32u;
    uint result = 0u;

    // For each destination bit in this word, find where it comes from in the source
    for (uint bit = 0u; bit < 32u; bit++) {
        uint dst_global_bit = word_idx * 32u + bit;
        uint src_global_bit = (dst_global_bit + total_bits - shift) % total_bits;
        uint src_word = src_global_bit / 32u;
        uint src_pos = src_global_bit % 32u;
        if ((data_in[src_word] & (1u << src_pos)) != 0u) {
            result |= (1u << bit);
        }
    }
    data_out[word_idx] = result;
}
