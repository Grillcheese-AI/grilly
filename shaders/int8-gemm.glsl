#version 450

// INT8 weight-only GEMM with FP32 accumulation
// Activations: fp32, Weights: int8 (packed 4 per uint32), Scales: fp32 per group
// RDNA2 optimized: FP32 accumulator prevents overflow/NaN

layout (local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

// Activations: (M, K) fp32
layout(set = 0, binding = 0) readonly buffer Activations {
    float activations[];
};

// INT8 weights packed as uint32: (N, K/4)
// Each uint32 contains 4 int8 weights
layout(set = 0, binding = 1) readonly buffer WeightsInt8 {
    uint weights_packed[];
};

// Per-group scales: (N, ceil(K / group_size))
layout(set = 0, binding = 2) readonly buffer Scales {
    float scales[];
};

// Output: (M, N) fp32
layout(set = 0, binding = 3) buffer Output {
    float output_data[];
};

layout(push_constant) uniform PushConsts {
    uint M;
    uint K;
    uint N;
    uint group_size;
};

// Unpack int8 from uint32 (little-endian)
float unpack_int8(uint packed, uint idx) {
    uint shift = idx * 8;
    uint byte_val = (packed >> shift) & 0xFFu;
    // Sign-extend: if bit 7 is set, it's negative
    int signed_val = int(byte_val);
    if (signed_val >= 128) signed_val -= 256;
    return float(signed_val);
}

void main() {
    uint col = gl_GlobalInvocationID.x;  // N index (output column)
    uint row = gl_GlobalInvocationID.y;  // M index (output row)

    if (row >= M || col >= N) return;

    uint num_groups = (K + group_size - 1) / group_size;
    uint packed_K = (K + 3) / 4;  // number of uint32s per row

    float sum = 0.0;

    for (uint k = 0; k < K; k++) {
        // Get activation value
        float act = activations[row * K + k];

        // Get packed weight and unpack
        uint pack_idx = k / 4;
        uint pack_offset = k % 4;
        uint packed = weights_packed[col * packed_K + pack_idx];
        float w = unpack_int8(packed, pack_offset);

        // Get per-group scale
        uint group_idx = k / group_size;
        float s = scales[col * num_groups + group_idx];

        // FP32 accumulate: act * (w_int8 * scale)
        sum += act * w * s;
    }

    output_data[row * N + col] = sum;
}
