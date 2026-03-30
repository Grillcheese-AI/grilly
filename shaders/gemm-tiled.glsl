#version 450

// Optimized Tiled GEMM: C[M,N] = A[M,K] @ B[N,K]^T + bias[N]
//
// RDNA 2 optimizations applied:
//   1. Bank conflict padding: shared[TILE][TILE+1] — eliminates 32-bank conflicts
//   2. 4×4 per-thread sub-tile: 16 FMAs per K step, high ALU utilization
//   3. 64 threads/workgroup = 1 Wave64 (perfect RDNA 2 occupancy)
//   4. Prefetch from LDS into registers before compute
//   5. Coalesced global loads: sequential threads read sequential memory
//
// Shared memory: 2 × 32 × 33 × 4 bytes = 8.4KB (fits in 64KB LDS)
// Compute: 32 K-steps × 16 FMAs = 512 FMAs per thread per tile

#define TILE 32
#define TT 4  // Thread tile: each thread computes 4×4 outputs

layout(local_size_x = 8, local_size_y = 8) in;

layout(std430, set = 0, binding = 0) readonly buffer InputBuf  { float A[]; };
layout(std430, set = 0, binding = 1) readonly buffer WeightBuf { float B[]; };
layout(std430, set = 0, binding = 2) readonly buffer BiasBuf   { float bias[]; };
layout(std430, set = 0, binding = 3) writeonly buffer OutputBuf { float C[]; };

layout(push_constant) uniform Params {
    uint M, K, N, has_bias;
};

// +1 padding eliminates bank conflicts on RDNA 2 (32 banks)
shared float tileA[TILE][TILE + 1];
shared float tileB[TILE][TILE + 1];

void main() {
    uint tileRow = gl_WorkGroupID.y * TILE;
    uint tileCol = gl_WorkGroupID.x * TILE;
    uint lx = gl_LocalInvocationID.x;  // 0..7
    uint ly = gl_LocalInvocationID.y;  // 0..7
    uint lid = ly * 8 + lx;            // 0..63

    // Accumulators for 4×4 output sub-tile
    float acc[TT][TT];
    for (int i = 0; i < TT; ++i)
        for (int j = 0; j < TT; ++j)
            acc[i][j] = 0.0;

    uint numTilesK = (K + TILE - 1) / TILE;

    for (uint tk = 0; tk < numTilesK; ++tk) {
        uint kOff = tk * TILE;

        // ── Cooperative tile load: 64 threads load 32×32 = 1024 elements ──
        // Each thread loads 16 elements (unrolled for compiler vectorization)
        for (uint i = lid; i < TILE * TILE; i += 64) {
            uint r = i >> 5;  // i / 32
            uint c = i & 31;  // i % 32

            // Load A[tileRow+r, kOff+c] — coalesced (sequential c)
            uint gR = tileRow + r;
            uint gC = kOff + c;
            tileA[r][c] = (gR < M && gC < K) ? A[gR * K + gC] : 0.0;

            // Load B^T[kOff+r, tileCol+c] = B[tileCol+c, kOff+r] — transposed
            uint gK = kOff + r;
            uint gN = tileCol + c;
            tileB[r][c] = (gK < K && gN < N) ? B[gN * K + gK] : 0.0;
        }

        barrier();

        // ── Compute 4×4 outer product per thread ──
        uint subRow = ly * TT;  // 0, 4, 8, 12, 16, 20, 24, 28
        uint subCol = lx * TT;

        for (uint k = 0; k < TILE; ++k) {
            // Prefetch from LDS into registers (padded → no bank conflicts)
            float a0 = tileA[subRow    ][k];
            float a1 = tileA[subRow + 1][k];
            float a2 = tileA[subRow + 2][k];
            float a3 = tileA[subRow + 3][k];

            float b0 = tileB[k][subCol    ];
            float b1 = tileB[k][subCol + 1];
            float b2 = tileB[k][subCol + 2];
            float b3 = tileB[k][subCol + 3];

            // 16 FMAs — compiler should schedule these to fill VALU pipelines
            acc[0][0] += a0 * b0;  acc[0][1] += a0 * b1;
            acc[0][2] += a0 * b2;  acc[0][3] += a0 * b3;
            acc[1][0] += a1 * b0;  acc[1][1] += a1 * b1;
            acc[1][2] += a1 * b2;  acc[1][3] += a1 * b3;
            acc[2][0] += a2 * b0;  acc[2][1] += a2 * b1;
            acc[2][2] += a2 * b2;  acc[2][3] += a2 * b3;
            acc[3][0] += a3 * b0;  acc[3][1] += a3 * b1;
            acc[3][2] += a3 * b2;  acc[3][3] += a3 * b3;
        }

        barrier();
    }

    // ── Write 4×4 results with bias ──
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
