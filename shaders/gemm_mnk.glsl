#version 450

/*
 * Simple tiled GEMM:
 *   C = A * B
 *
 * A: (M, K) row-major
 * B: (K, N) row-major
 * C: (M, N) row-major
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

shared float Asub[16][16]; // [row][k-tile]
shared float Bsub[16][16]; // [k-tile][col]

void main() {
    uint local_x = gl_LocalInvocationID.x; // 0..15
    uint local_y = gl_LocalInvocationID.y; // 0..15

    uint global_m = gl_WorkGroupID.y * 16u + local_y;
    uint global_n = gl_WorkGroupID.x * 16u + local_x;

    if (global_m >= params.M || global_n >= params.N) {
        // We still need to participate in barriers if we use them later,
        // but here we early return as we never read/write C.
        return;
    }

    float acc = 0.0;

    // Number of tiles along K
    uint num_tiles = (params.K + 16u - 1u) / 16u;

    for (uint tile_idx = 0u; tile_idx < num_tiles; tile_idx++) {
        // Starting k index for this tile
        uint k_base = tile_idx * 16u;

        // Load A tile: Asub[local_y][local_x] = A[global_m, k_base + local_x]
        uint kA = k_base + local_x;
        if (kA < params.K) {
            uint A_idx = global_m * params.K + kA;
            Asub[local_y][local_x] = A[A_idx];
        } else {
            Asub[local_y][local_x] = 0.0;
        }

        // Load B tile: Bsub[local_y][local_x] = B[k_base + local_y, global_n]
        uint kB = k_base + local_y;
        if (kB < params.K) {
            uint B_idx = kB * params.N + global_n;
            Bsub[local_y][local_x] = B[B_idx];
        } else {
            Bsub[local_y][local_x] = 0.0;
        }

        barrier();

        // Multiply this tile
        for (uint kk = 0u; kk < 16u; kk++) {
            acc += Asub[local_y][kk] * Bsub[kk][local_x];
        }

        barrier();
    }

    // Write C
    uint C_idx = global_m * params.N + global_n;
    C[C_idx] = acc;
}
