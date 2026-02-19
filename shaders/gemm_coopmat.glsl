#version 450
#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

/*
 * Cooperative Matrix GEMM: C = A * B
 *
 * Uses VK_KHR_cooperative_matrix for hardware-accelerated 16x16x16 matmul.
 * A: (M, K) row-major, float16
 * B: (K, N) row-major, float16
 * C: (M, N) row-major, float32 (mixed-precision accumulation)
 *
 * Each workgroup dispatches 4 subgroups (wave64 on RDNA2).
 * Each subgroup computes one 16x16 output tile via coopMatMulAdd.
 * Workgroup covers 16 rows x 64 cols of C.
 *
 * Inputs must be padded to multiples of 16 on CPU side.
 */

layout(local_size_x = 64, local_size_y = 4, local_size_z = 1) in;

layout(binding = 0) readonly buffer ABuffer { float16_t A[]; };
layout(binding = 1) readonly buffer BBuffer { float16_t B[]; };
layout(binding = 2) writeonly buffer CBuffer { float C[]; };

layout(push_constant) uniform PushConstants {
    uint M;  // rows of A / C (padded to multiple of 16)
    uint K;  // cols of A / rows of B (padded to multiple of 16)
    uint N;  // cols of B / C (padded to multiple of 16)
} params;

void main() {
    // Each subgroup computes a 16x16 tile of C.
    // gl_LocalInvocationID.y selects which of the 4 subgroups (0..3).
    uint tile_row = gl_WorkGroupID.y * 16u;
    uint tile_col = gl_WorkGroupID.x * 64u + gl_LocalInvocationID.y * 16u;

    // Declare cooperative matrices (16x16 tiles)
    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> matA;
    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> matB;
    coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> matC =
        coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);

    // Tile over K dimension in steps of 16
    for (uint k = 0u; k < params.K; k += 16u) {
        // Load A tile: rows [tile_row..+16), cols [k..+16)
        coopMatLoad(matA, A, tile_row * params.K + k, params.K,
                    gl_CooperativeMatrixLayoutRowMajor);

        // Load B tile: rows [k..+16), cols [tile_col..+16)
        coopMatLoad(matB, B, k * params.N + tile_col, params.N,
                    gl_CooperativeMatrixLayoutRowMajor);

        // Hardware multiply-accumulate: matC += matA * matB
        matC = coopMatMulAdd(matA, matB, matC);
    }

    // Store 16x16 result tile to C
    coopMatStore(matC, C, tile_row * params.N + tile_col, params.N,
                 gl_CooperativeMatrixLayoutRowMajor);
}
