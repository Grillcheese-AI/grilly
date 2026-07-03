#version 450

// Ternary weight-only GEMM (BitNet b1.58), MULTIPLY-FREE inner loop.
//
//   output[m,n] = alpha * sum_k activation[m,k] * trit[n,k]
//   trit in {-1, 0, +1};  alpha = mean|W| (ONE per-tensor scale, applied once)
//
// The point vs int8-gemm.glsl: int8 saves weight BYTES but still does an fp32
// multiply per element. Ternary makes the multiply DISAPPEAR — w*a with
// w in {-1,0,+1} is "add a", "subtract a", or "skip". The hot loop is a
// conditional-negate + masked accumulate, no multiply. The single alpha scale
// multiplies the finished sum exactly once per output element.
//
// PACKING (v1, 2-bit): 16 trits per uint32, 2 bits each, encoded
//   0b00 -> 0,  0b01 -> +1,  0b10 -> -1   (0b11 unused).
// 8x fewer weight bytes than fp32. The 1.58-bit packing (5 trits / 8 bits,
// 3^5=243<256) is a later swap — it changes ONLY unpack_trit() and the
// host packer, never the accumulation below. That is the whole design seam:
// the inner loop consumes trits from unpack_trit(), format-agnostic.
//
// Weights are (N, K) laid out row-major, packed along K: each row is
// ceil(K/16) uint32s. Activations fp32 (M, K). Output fp32 (M, N).
//
// RDNA2 note: fp32 accumulator (no overflow worry at these K); the
// conditional-negate lowers to a sign-flip the ALU does for free. Whether
// that beats a straight FMA in ENERGY is exactly what the J/token harness
// measures — the guaranteed win is the 8x byte reduction (memory-bound
// decode), the possible extra win is the op-level multiply-free property.

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer Activations {
    float activations[];        // (M, K)
};

// Packed ternary weights: (N, ceil(K/16)) uint32, 16 trits/word, 2 bits each.
layout(set = 0, binding = 1) readonly buffer WeightsTernary {
    uint weights_packed[];
};

layout(set = 0, binding = 2) buffer Output {
    float output_data[];        // (M, N)
};

layout(push_constant) uniform PushConsts {
    uint M;
    uint K;
    uint N;
    float alpha;                // per-tensor scale = mean|W|
};

const uint TRITS_PER_WORD = 16u;   // v1 2-bit packing

// Decode one trit in {-1,0,+1} from a packed word. THE SEAM: swap this (and
// the host packer) for 1.58-bit later; the accumulator never changes.
//   2-bit code: 00 -> 0, 01 -> +1, 10 -> -1
int unpack_trit(uint word, uint idx) {
    uint code = (word >> (idx * 2u)) & 0x3u;
    // 0->0, 1->+1, 2->-1  (branchless: map via small arithmetic)
    //   code==1 -> +1 ; code==2 -> -1 ; else 0
    return int(code == 1u) - int(code == 2u);
}

void main() {
    uint col = gl_GlobalInvocationID.x;   // N
    uint row = gl_GlobalInvocationID.y;   // M
    if (row >= M || col >= N) return;

    uint words_per_row = (K + TRITS_PER_WORD - 1u) / TRITS_PER_WORD;
    uint abase = row * K;
    uint wbase = col * words_per_row;

    // MULTIPLY-FREE accumulation: only add / subtract / skip.
    float pos = 0.0;   // sum of activations where trit == +1
    float neg = 0.0;   // sum of activations where trit == -1

    uint k = 0u;
    for (uint w = 0u; w < words_per_row; ++w) {
        uint word = weights_packed[wbase + w];
        // 16 trits in this word (last word may run past K — guarded by k<K)
        for (uint t = 0u; t < TRITS_PER_WORD && k < K; ++t, ++k) {
            int tr = unpack_trit(word, t);
            float a = activations[abase + k];
            // conditional-negate + mask, no multiply:
            //   tr==+1 -> pos += a ; tr==-1 -> neg += a ; tr==0 -> nothing
            if (tr > 0) pos += a;
            else if (tr < 0) neg += a;
        }
    }

    // ONE multiply per output element: the per-tensor scale.
    output_data[row * N + col] = alpha * (pos - neg);
}
