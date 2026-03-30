#version 450

// Tiled GEMM: C[M,N] = A[M,K] @ B[N,K]^T + bias[N]
//
// Uses shared memory (LDS) tiling for data reuse:
//   - Each workgroup computes a TILE_M × TILE_N block of the output
//   - Tiles of A and B are loaded cooperatively into shared memory
//   - Each thread accumulates TILE_M/WG_Y × TILE_N/WG_X output elements
//
// For SigLIP2 fixed shapes:
//   QKV:  M=1024, K=768, N=2304  (the biggest matmul)
//   fc1:  M=1024, K=768, N=3072
//   fc2:  M=1024, K=3072, N=768
//   out:  M=1024, K=768, N=768
//
// Tuned for RDNA 2 (RX 6750 XT):
//   - TILE = 32×32, local_size = 16×16
//   - Each thread computes a 2×2 sub-tile
//   - 256 threads per workgroup = 4 Wave64s = good occupancy
//   - Shared memory: 2 × 32 × 32 × 4 bytes = 8KB (fits in 64KB LDS)

#define TILE 32
#define THREAD_TILE 2  // Each thread computes 2×2 output elements

layout(local_size_x = 16, local_size_y = 16) in;

layout(std430, set = 0, binding = 0) readonly buffer InputBuf  { float A[]; };
layout(std430, set = 0, binding = 1) readonly buffer WeightBuf { float B[]; }; // B[N,K] row-major
layout(std430, set = 0, binding = 2) readonly buffer BiasBuf   { float bias[]; };
layout(std430, set = 0, binding = 3) writeonly buffer OutputBuf { float C[]; };

layout(push_constant) uniform Params {
    uint M;         // rows of A (seq_len)
    uint K;         // cols of A = cols of B (input_dim)
    uint N;         // rows of B = output_dim
    uint has_bias;
};

shared float tileA[TILE][TILE];   // Shared memory tile for A
shared float tileB[TILE][TILE];   // Shared memory tile for B^T

void main() {
    // Which output tile this workgroup computes
    uint tileRow = gl_WorkGroupID.y * TILE;
    uint tileCol = gl_WorkGroupID.x * TILE;

    // Thread position within the workgroup
    uint lx = gl_LocalInvocationID.x;  // 0..15
    uint ly = gl_LocalInvocationID.y;  // 0..15

    // Each thread accumulates a 2×2 sub-tile of the output
    float acc00 = 0.0, acc01 = 0.0;
    float acc10 = 0.0, acc11 = 0.0;

    // Number of tiles along the K dimension
    uint numTilesK = (K + TILE - 1) / TILE;

    for (uint tk = 0; tk < numTilesK; ++tk) {
        uint kOffset = tk * TILE;

        // Cooperatively load tile of A[tileRow..+TILE, kOffset..+TILE]
        // 256 threads load 32×32 = 1024 elements → 4 elements per thread
        for (uint i = 0; i < TILE * TILE; i += 256) {
            uint idx = i + ly * 16 + lx;
            if (idx < TILE * TILE) {
                uint r = idx / TILE;
                uint c = idx % TILE;
                uint globalR = tileRow + r;
                uint globalC = kOffset + c;
                tileA[r][c] = (globalR < M && globalC < K) ? A[globalR * K + globalC] : 0.0;
            }
        }

        // Load tile of B^T: B is stored as B[N,K], we need B^T[K,N]
        // So tileB[k][n] = B[tileCol + n][kOffset + k]
        for (uint i = 0; i < TILE * TILE; i += 256) {
            uint idx = i + ly * 16 + lx;
            if (idx < TILE * TILE) {
                uint r = idx / TILE;  // k index
                uint c = idx % TILE;  // n index
                uint globalK = kOffset + r;
                uint globalN = tileCol + c;
                tileB[r][c] = (globalK < K && globalN < N) ? B[globalN * K + globalK] : 0.0;
            }
        }

        barrier();

        // Compute: each thread does a 2×2 sub-tile
        uint subRow0 = ly * 2;
        uint subCol0 = lx * 2;

        for (uint k = 0; k < TILE; ++k) {
            float a0 = tileA[subRow0][k];
            float a1 = tileA[subRow0 + 1][k];
            float b0 = tileB[k][subCol0];
            float b1 = tileB[k][subCol0 + 1];

            acc00 += a0 * b0;
            acc01 += a0 * b1;
            acc10 += a1 * b0;
            acc11 += a1 * b1;
        }

        barrier();
    }

    // Write results with bias
    uint r0 = tileRow + ly * 2;
    uint c0 = tileCol + lx * 2;

    if (r0 < M && c0 < N) {
        C[r0 * N + c0] = acc00 + (has_bias != 0 ? bias[c0] : 0.0);
    }
    if (r0 < M && c0 + 1 < N) {
        C[r0 * N + c0 + 1] = acc01 + (has_bias != 0 ? bias[c0 + 1] : 0.0);
    }
    if (r0 + 1 < M && c0 < N) {
        C[(r0 + 1) * N + c0] = acc10 + (has_bias != 0 ? bias[c0] : 0.0);
    }
    if (r0 + 1 < M && c0 + 1 < N) {
        C[(r0 + 1) * N + c0 + 1] = acc11 + (has_bias != 0 ? bias[c0 + 1] : 0.0);
    }
}
