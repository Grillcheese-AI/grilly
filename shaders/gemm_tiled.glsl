#version 450

/*
 * Optimized tiled GEMM with 4x4 register blocking:
 *   C = A * B
 *
 * A: (M, K) row-major
 * B: (K, N) row-major
 * C: (M, N) row-major
 *
 * Each 16x16 workgroup computes a 64x64 output tile.
 * Each thread computes a 4x4 block of C using register accumulators.
 * Shared memory tiles: 64 x 16 for A and 16 x 64 for B.
 */

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(binding = 0) readonly buffer ABuffer {
    float A[];
};

layout(binding = 1) readonly buffer BBuffer {
    float B[];
};

layout(binding = 2) writeonly buffer CBuffer {
    float C[];
};

layout(push_constant) uniform PushConstants {
    uint M; // rows of A, rows of C
    uint K; // cols of A, rows of B
    uint N; // cols of B, cols of C
} params;

// Shared memory: 64 rows x 16 cols for A tile, 16 rows x 64 cols for B tile
shared float Asub[64][16];
shared float Bsub[16][65];  // +1 padding to avoid shared memory bank conflicts

void main() {
    uint lx = gl_LocalInvocationID.x; // 0..15
    uint ly = gl_LocalInvocationID.y; // 0..15

    // Each workgroup covers a 64x64 output tile
    uint tile_row = gl_WorkGroupID.y * 64u;
    uint tile_col = gl_WorkGroupID.x * 64u;

    // Each thread computes 4 rows x 4 cols of C
    // Thread (lx, ly) handles rows [ly*4 .. ly*4+3] and cols [lx*4 .. lx*4+3]
    float acc[4][4];
    for (uint i = 0u; i < 4u; i++)
        for (uint j = 0u; j < 4u; j++)
            acc[i][j] = 0.0;

    uint num_tiles = (params.K + 16u - 1u) / 16u;

    for (uint t = 0u; t < num_tiles; t++) {
        uint k_base = t * 16u;

        // Load A tile: 64 rows x 16 cols
        // 256 threads load 64*16 = 1024 elements (4 elements per thread)
        uint linear_id = ly * 16u + lx; // 0..255
        for (uint i = 0u; i < 4u; i++) {
            uint idx = linear_id + i * 256u;
            uint a_row = tile_row + (idx / 16u);
            uint a_col = k_base + (idx % 16u);
            Asub[idx / 16u][idx % 16u] =
                (a_row < params.M && a_col < params.K)
                ? A[a_row * params.K + a_col] : 0.0;
        }

        // Load B tile: 16 rows x 64 cols
        // 256 threads load 16*64 = 1024 elements (4 elements per thread)
        for (uint i = 0u; i < 4u; i++) {
            uint idx = linear_id + i * 256u;
            uint b_row = k_base + (idx / 64u);
            uint b_col = tile_col + (idx % 64u);
            Bsub[idx / 64u][idx % 64u] =
                (b_row < params.K && b_col < params.N)
                ? B[b_row * params.N + b_col] : 0.0;
        }

        barrier();

        // Compute: each thread accumulates 4x4 block
        for (uint kk = 0u; kk < 16u; kk++) {
            // Load 4 values from A column and 4 from B row into registers
            float a0 = Asub[ly * 4u + 0u][kk];
            float a1 = Asub[ly * 4u + 1u][kk];
            float a2 = Asub[ly * 4u + 2u][kk];
            float a3 = Asub[ly * 4u + 3u][kk];

            float b0 = Bsub[kk][lx * 4u + 0u];
            float b1 = Bsub[kk][lx * 4u + 1u];
            float b2 = Bsub[kk][lx * 4u + 2u];
            float b3 = Bsub[kk][lx * 4u + 3u];

            acc[0][0] += a0 * b0;
            acc[0][1] += a0 * b1;
            acc[0][2] += a0 * b2;
            acc[0][3] += a0 * b3;

            acc[1][0] += a1 * b0;
            acc[1][1] += a1 * b1;
            acc[1][2] += a1 * b2;
            acc[1][3] += a1 * b3;

            acc[2][0] += a2 * b0;
            acc[2][1] += a2 * b1;
            acc[2][2] += a2 * b2;
            acc[2][3] += a2 * b3;

            acc[3][0] += a3 * b0;
            acc[3][1] += a3 * b1;
            acc[3][2] += a3 * b2;
            acc[3][3] += a3 * b3;
        }

        barrier();
    }

    // Write 4x4 block to C
    for (uint i = 0u; i < 4u; i++) {
        uint c_row = tile_row + ly * 4u + i;
        if (c_row >= params.M) continue;
        for (uint j = 0u; j < 4u; j++) {
            uint c_col = tile_col + lx * 4u + j;
            if (c_col >= params.N) continue;
            C[c_row * params.N + c_col] = acc[i][j];
        }
    }
}
