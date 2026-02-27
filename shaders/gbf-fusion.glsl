#version 450
#extension GL_KHR_shader_subgroup_arithmetic : require

// ── Geometric Bottleneck Fusion (GBF) & Explain Away ────────────────────
//
// Operates on bitpacked bipolar vectors in working memory.
//
// Mode 0 — FUSE:  bundle_bits ^= operand_bits
//   XOR-binds an operand (text, vision, audio) into the working bundle.
//   In bipolar VSA, XOR binding is its own inverse and runs in O(1).
//
// Mode 1 — EXPLAIN_AWAY:  bundle_bits ^= operand_bits
//   XOR-unbinds a decoded winner from the bundle, suppressing its echo
//   so the next resonator search finds the next-most-similar entry.
//   Mathematically identical to FUSE (XOR is self-inverse), but semantically
//   different: FUSE adds signal, EXPLAIN_AWAY removes it.
//
// Mode 2 — ACCUMULATE:  bundle_bits |= operand_bits
//   OR-based accumulation for majority-vote bundling of multiple modalities.
//   Use when fusing more than 2 modalities before final thresholding.
//
// Dispatch: groupCountX = 1, groupCountY = 1, groupCountZ = 1
//   Single workgroup processes the full vector (256 threads x loop).
//   At d_vsa=10240: 320 uint32 words, ~2 iterations per thread.
//
// Buffers:
//   binding 0 — WorkingMemory (read/write): the multimodal bundle state
//   binding 1 — Operand (read-only): the vector to fuse/subtract
//
// Push constants:
//   words_per_vec: uint32 words in the bitpacked vector (dim/32)
//   mode: 0=FUSE, 1=EXPLAIN_AWAY, 2=ACCUMULATE
// ────────────────────────────────────────────────────────────────────────

layout(local_size_x = 256) in;

layout(std430, set = 0, binding = 0) buffer WorkingMemory {
    uint bundle_bits[];
};

layout(std430, set = 0, binding = 1) readonly buffer Operand {
    uint operand_bits[];
};

layout(push_constant) uniform PushConsts {
    uint words_per_vec;  // e.g., 320 for 10240d
    uint mode;           // 0 = FUSE, 1 = EXPLAIN_AWAY, 2 = ACCUMULATE
};

void main() {
    uint tid = gl_LocalInvocationID.x;

    for (uint i = tid; i < words_per_vec; i += gl_WorkGroupSize.x) {
        uint bundle_word = bundle_bits[i];
        uint operand_word = operand_bits[i];

        if (mode == 0 || mode == 1) {
            // FUSE or EXPLAIN_AWAY: XOR binding (self-inverse)
            bundle_bits[i] = bundle_word ^ operand_word;
        }
        else if (mode == 2) {
            // ACCUMULATE: OR-based bundling
            bundle_bits[i] = bundle_word | operand_word;
        }
    }

    // Ensure writes are visible before next dispatch reads the buffer
    memoryBarrierBuffer();
}
