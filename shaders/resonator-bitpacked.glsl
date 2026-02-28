#version 450

// ── Bitpacked Hamming Resonator ──────────────────────────────────────
//
// Calculates the VSA Similarity for every entry in a word codebook:
//   similarity = 1.0 - (2.0 * hamming_distance / dim)
//
// Maps Hamming distance [0, dim] to bipolar cosine similarity [-1, +1].
// Uses bitCount(XOR) which is a single-cycle hardware instruction on
// AMD RDNA 2/3, replacing expensive float32 dot products entirely.
//
// Dispatch: workgroups_x = codebook_size (one workgroup per word)
//           Each workgroup's 256 threads cooperatively reduce the
//           Hamming distance across all uint32 words of the vector.
//
// Buffer layout:
//   binding 0: query_packed[]       — bitpacked query vector (words_per_vec uint32s)
//   binding 1: codebook_packed[]    — bitpacked word codebook (codebook_size * words_per_vec)
//   binding 2: similarities[]       — output float scores [-1.0, 1.0]
//

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer QueryBuf {
    uint query_packed[];
};

layout(set = 0, binding = 1) readonly buffer CodebookBuf {
    uint codebook_packed[];
};

layout(set = 0, binding = 2) writeonly buffer SimilarityBuf {
    float similarities[];
};

layout(push_constant) uniform PushConsts {
    uint dim;             // Original bipolar dimension (e.g., 10240)
    uint words_per_vec;   // dim / 32 (e.g., 320)
    uint codebook_size;   // Number of entries in the codebook
    uint _pad;            // Alignment to 16 bytes
};

// LDS for parallel reduction of partial Hamming distances
shared uint partial_hamming[256];

void main() {
    uint vec_idx = gl_WorkGroupID.x;
    uint local_idx = gl_LocalInvocationID.x;

    if (vec_idx >= codebook_size) return;

    // ── Phase 1: Parallel Hamming Distance ───────────────────────────
    // Each of the 256 threads processes a strided subset of the vector.
    // At words_per_vec=320, each thread handles ~1-2 words.
    // bitCount(XOR) is a single VALU cycle on RDNA 2.
    uint hamming_dist = 0;
    uint offset = vec_idx * words_per_vec;

    for (uint i = local_idx; i < words_per_vec; i += 256) {
        uint q = query_packed[i];
        uint c = codebook_packed[offset + i];
        hamming_dist += bitCount(q ^ c);
    }

    partial_hamming[local_idx] = hamming_dist;
    barrier();

    // ── Phase 2: LDS Parallel Tree Reduction ─────────────────────────
    // Reduces 256 partial sums to a single total in O(log N) steps.
    // The barrier() between steps ensures memory consistency.
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (local_idx < stride) {
            partial_hamming[local_idx] += partial_hamming[local_idx + stride];
        }
        barrier();
    }

    // ── Phase 3: Similarity Mapping ──────────────────────────────────
    // Thread 0 writes the final similarity score.
    //
    // In bipolar VSA, the dot product <a,b> = dim - 2*hamming(a,b).
    // Normalizing: similarity = (dim - 2*hamming) / dim
    //            = 1.0 - 2.0 * hamming / dim
    //
    // Range: [+1.0 = identical, 0.0 = orthogonal, -1.0 = anti-correlated]
    if (local_idx == 0) {
        uint total_hamming = partial_hamming[0];
        float sim = (float(dim) - 2.0 * float(total_hamming)) / float(dim);
        similarities[vec_idx] = sim;
    }
}
