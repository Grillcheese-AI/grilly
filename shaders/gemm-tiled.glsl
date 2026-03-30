#version 450

// Optimized Tiled GEMM: C[M,N] = A[M,K] @ B[N,K]^T + bias[N]
//
// RDNA 2 optimizations:
//   1. Bank conflict padding on BOTH tiles: [TILE][TILE+1]
//   2. 4×4 per-thread sub-tile: 16 FMAs per K step
//   3. 64 threads = 1 Wave64 (perfect RDNA 2 occupancy)
//   4. Explicit fma() intrinsic — forces hardware FMA (1 cycle vs 2)
//   5. No bounds checking in inner loop — caller pads buffers to TILE multiples
//   6. Register prefetch from LDS before compute

#define TILE 32
#define TT 4

layout(local_size_x = 8, local_size_y = 8) in;

layout(std430, set = 0, binding = 0) readonly buffer InputBuf  { float A[]; };
layout(std430, set = 0, binding = 1) readonly buffer WeightBuf { float B[]; };
layout(std430, set = 0, binding = 2) readonly buffer BiasBuf   { float bias[]; };
layout(std430, set = 0, binding = 3) writeonly buffer OutputBuf { float C[]; };

layout(push_constant) uniform Params {
    uint M, K, N, has_bias;
    uint M_padded, K_padded, N_padded;  // Padded dimensions (multiples of TILE)
};

// Bank conflict padding on BOTH tiles (+1 on inner dimension)
shared float tileA[TILE][TILE + 1];
shared float tileB[TILE][TILE + 1];

void main() {
    uint tileRow = gl_WorkGroupID.y * TILE;
    uint tileCol = gl_WorkGroupID.x * TILE;
    uint lx = gl_LocalInvocationID.x;
    uint ly = gl_LocalInvocationID.y;
    uint lid = ly * 8 + lx;

    float acc[TT][TT];
    for (int i = 0; i < TT; ++i)
        for (int j = 0; j < TT; ++j)
            acc[i][j] = 0.0;

    // K_padded is guaranteed to be a multiple of TILE by the caller
    uint numTilesK = K_padded >> 5;  // K_padded / 32

    for (uint tk = 0; tk < numTilesK; ++tk) {
        uint kOff = tk * TILE;

        // ── Cooperative tile load — NO bounds checks (buffers are padded) ──
        // 64 threads load 32×32 = 1024 elements, 16 per thread
        for (uint i = lid; i < TILE * TILE; i += 64) {
            uint r = i >> 5;    // i / 32
            uint c = i & 31;    // i % 32

            // A tile: coalesced sequential reads
            tileA[r][c] = A[(tileRow + r) * K_padded + kOff + c];

            // B^T tile: B[tileCol+c, kOff+r] transposed
            tileB[r][c] = B[(tileCol + c) * K_padded + kOff + r];
        }

        barrier();

        // ── 4×4 outer product with explicit fma() ──
        uint subRow = ly * TT;
        uint subCol = lx * TT;

        for (uint k = 0; k < TILE; ++k) {
            // Register prefetch from padded LDS (zero bank conflicts)
            float a0 = tileA[subRow    ][k];
            float a1 = tileA[subRow + 1][k];
            float a2 = tileA[subRow + 2][k];
            float a3 = tileA[subRow + 3][k];

            float b0 = tileB[k][subCol    ];
            float b1 = tileB[k][subCol + 1];
            float b2 = tileB[k][subCol + 2];
            float b3 = tileB[k][subCol + 3];

            // 16 hardware FMAs — 1 cycle each on RDNA 2 VALU
            acc[0][0] = fma(a0, b0, acc[0][0]);
            acc[0][1] = fma(a0, b1, acc[0][1]);
            acc[0][2] = fma(a0, b2, acc[0][2]);
            acc[0][3] = fma(a0, b3, acc[0][3]);
            acc[1][0] = fma(a1, b0, acc[1][0]);
            acc[1][1] = fma(a1, b1, acc[1][1]);
            acc[1][2] = fma(a1, b2, acc[1][2]);
            acc[1][3] = fma(a1, b3, acc[1][3]);
            acc[2][0] = fma(a2, b0, acc[2][0]);
            acc[2][1] = fma(a2, b1, acc[2][1]);
            acc[2][2] = fma(a2, b2, acc[2][2]);
            acc[2][3] = fma(a2, b3, acc[2][3]);
            acc[3][0] = fma(a3, b0, acc[3][0]);
            acc[3][1] = fma(a3, b1, acc[3][1]);
            acc[3][2] = fma(a3, b2, acc[3][2]);
            acc[3][3] = fma(a3, b3, acc[3][3]);
        }

        barrier();
    }

    // ── Write results (bounds check only on output, not inner loop) ──
    for (int i = 0; i < TT; ++i) {
        uint r = tileRow + ly * TT + i;
        if (r >= M) continue;
        for (int j = 0; j < TT; ++j) {
            uint c = tileCol + lx * TT + j;
            if (c >= N) continue;
            float val = acc[i][j];
            if (has_bias != 0) val += bias[c];
            C[r * N + c] = val;
        }
    }
}
