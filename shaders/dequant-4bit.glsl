#version 450

// 4-bit block-quantized GEMM with FP32 accumulation
// Weights: 4-bit unsigned packed 8 per uint32
// Per-block scale + zero-point for accurate dequantization

layout (local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

// Activations: (M, K) fp32
layout(set = 0, binding = 0) readonly buffer Activations {
    float activations[];
};

// 4-bit weights packed as uint32: (N, K/8)
// Each uint32 contains 8 x 4-bit weights
layout(set = 0, binding = 1) readonly buffer Weights4Bit {
    uint weights_packed[];
};

// Per-block scales: (N, num_blocks) fp32
layout(set = 0, binding = 2) readonly buffer Scales {
    float scales[];
};

// Per-block zero points: (N, num_blocks) fp32
layout(set = 0, binding = 3) readonly buffer Zeros {
    float zeros[];
};

// Output: (M, N) fp32
layout(set = 0, binding = 4) buffer Output {
    float output_data[];
};

layout(push_constant) uniform PushConsts {
    uint M;
    uint K;
    uint N;
    uint block_size;
};

// Unpack 4-bit value from uint32 (unsigned, 0-15)
float unpack_4bit(uint packed, uint idx) {
    uint shift = idx * 4;
    uint nibble = (packed >> shift) & 0xFu;
    return float(nibble);
}

void main() {
    uint col = gl_GlobalInvocationID.x;  // N index
    uint row = gl_GlobalInvocationID.y;  // M index

    if (row >= M || col >= N) return;

    uint num_blocks = (K + block_size - 1) / block_size;
    uint packed_K = (K + 7) / 8;  // uint32s per weight row

    float sum = 0.0;

    for (uint k = 0; k < K; k++) {
        float act = activations[row * K + k];

        // Unpack 4-bit weight
        uint pack_idx = k / 8;
        uint pack_offset = k % 8;
        uint packed = weights_packed[col * packed_K + pack_idx];
        float w_quant = unpack_4bit(packed, pack_offset);

        // Dequantize: w_fp32 = (w_quant - zero_point) * scale
        uint block_idx = k / block_size;
        float s = scales[col * num_blocks + block_idx];
        float z = zeros[col * num_blocks + block_idx];
        float w = (w_quant - z) * s;

        sum += act * w;
    }

    output_data[row * N + col] = sum;
}
