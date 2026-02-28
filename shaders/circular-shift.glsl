#version 450

// Temporal Permutation Shader: Batched circular bit shift for time binding.
//
// Shifts N bitpacked VSA vectors (10240 bits = 320 uint32 words each)
// by an arbitrary number of bits in a single parallel dispatch.
//
// Dispatch model:
//   vkCmdDispatch(batch_size, 1, 1)
//   Each workgroup = 1 timeline (320 threads, 1 per uint32 word)
//   gl_WorkGroupID.x indexes which timeline in the contiguous buffer
//
// The input buffer contains batch_size vectors laid out contiguously:
//   [timeline_0: 320 words][timeline_1: 320 words]...[timeline_N: 320 words]
//
// Modes:
//   mode 0: Circular RIGHT shift (bind_time)
//   mode 1: Circular LEFT shift  (unbind_time)

layout(local_size_x = 320, local_size_y = 1, local_size_z = 1) in;

layout(std430, set = 0, binding = 0) readonly buffer InputState {
    uint in_bits[];
};

layout(std430, set = 0, binding = 1) writeonly buffer OutputState {
    uint out_bits[];
};

layout(push_constant) uniform PushConsts {
    uint words_per_vec;  // 320 for 10240d
    uint shift_amount;   // T value (number of bits to shift)
    uint mode;           // 0 = right shift (bind), 1 = left shift (unbind)
};

void main() {
    uint idx = gl_LocalInvocationID.x;
    if (idx >= words_per_vec) return;

    // Each workgroup handles one timeline
    uint base_offset = gl_WorkGroupID.x * words_per_vec;

    // For left shift, invert the direction
    uint effective_shift = shift_amount;
    if (mode == 1) {
        effective_shift = (words_per_vec * 32) - shift_amount;
    }

    // Macro (word-level) and micro (bit-level) decomposition
    uint word_shift = effective_shift / 32;
    uint bit_shift  = effective_shift % 32;

    // Source indices with circular wrap-around
    uint src_idx  = (idx + words_per_vec - word_shift) % words_per_vec;
    uint prev_idx = (src_idx + words_per_vec - 1) % words_per_vec;

    // Read source words from this timeline's slice
    uint current_word = in_bits[base_offset + src_idx];
    uint prev_word    = in_bits[base_offset + prev_idx];

    // Bitwise shift and combine carry bits
    uint shifted_val;
    if (bit_shift == 0) {
        shifted_val = current_word;
    } else {
        shifted_val = (current_word >> bit_shift)
                    | (prev_word << (32 - bit_shift));
    }

    out_bits[base_offset + idx] = shifted_val;
}
