#version 450
#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_KHR_shader_subgroup_basic : require

/*
 * Cooperative Matrix GEMM with Shared Memory Staging ("The Merge")
 *
 * C = A * B    where A is MxK, B is KxN, C is MxN, row-major.
 * A and B are float16; C accumulates in float32.
 *
 * Workgroup: 64x4 (256 threads) -> 4 subgroups of 64 lanes (Wave64 for RDNA).
 * Output tile per workgroup: 16 rows x 64 cols of C (4 x 16x16 coopmat tiles).
 *
 * Alignment requirements on the caller:
 *     M % 16 == 0, K % 16 == 0, N % 64 == 0
 *
 * Dispatch:
 *     gx = N / 64
 *     gy = M / 16
 *     gz = 1
 *
 * Hardware notes: on RDNA3 / NVIDIA Tensor Cores this hits full WMMA
 * throughput via the driver. On RDNA1/RDNA2 it runs through the driver's
 * emulation path (standard fp16 vector ops) — still correct, noticeably
 * slower than the peak but still competitive with a hand-tuled GEMM.
 */

layout(local_size_x = 64, local_size_y = 4, local_size_z = 1) in;

layout(binding = 0) readonly buffer ABuffer { float16_t A[]; };
layout(binding = 1) readonly buffer BBuffer { float16_t B[]; };
layout(binding = 2) writeonly buffer CBuffer { float C[]; };

layout(push_constant) uniform PushConstants {
    uint M;
    uint K;
    uint N;
} params;

// Shared memory staging:
//   Asub stages a 16x16 tile of A (256 elements, all 4 subgroups share it)
//   Bsub stages a 16x64 tile of B (1024 elements, each subgroup takes a slice)
shared float16_t Asub[256];
shared float16_t Bsub[1024];

void main() {
    uint tile_row = gl_WorkGroupID.y * 16u;
    uint tile_col = gl_WorkGroupID.x * 64u;

    uint sg_id   = gl_LocalInvocationID.y;  // subgroup id 0..3
    uint lane_id = gl_LocalInvocationID.x;  // lane within subgroup 0..63
    uint linear_id = sg_id * 64u + lane_id; // 0..255

    // Hardware accumulator lives in registers (one per subgroup)
    coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> matC =
        coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);

    for (uint k = 0u; k < params.K; k += 16u) {
        // ── 1. Stage A tile (16x16 = 256 elements, one per thread) ──
        if (linear_id < 256u) {
            uint a_r = linear_id / 16u;
            uint a_c = linear_id % 16u;
            Asub[linear_id] = A[(tile_row + a_r) * params.K + (k + a_c)];
        }

        // ── 2. Stage B tile (16x64 = 1024 elements, 4 per thread) ──
        for (uint i = 0u; i < 4u; ++i) {
            uint b_idx = linear_id + (i * 256u);
            uint b_r   = b_idx / 64u;
            uint b_c   = b_idx % 64u;
            Bsub[b_idx] = B[(k + b_r) * params.N + (tile_col + b_c)];
        }

        barrier();

        // ── 3. Load shared memory → cooperative registers ──
        coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> matA;
        coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> matB;

        // All 4 subgroups load the SAME 16x16 A tile from Asub (stride = 16)
        coopMatLoad(matA, Asub, 0, 16, gl_CooperativeMatrixLayoutRowMajor);

        // Each subgroup loads its OWN 16x16 slice of the 16x64 B tile
        // (stride = 64 over Bsub, starting offset = sg_id * 16)
        coopMatLoad(matB, Bsub, sg_id * 16u, 64, gl_CooperativeMatrixLayoutRowMajor);

        // ── 4. Hardware matrix multiply-accumulate ──
        matC = coopMatMulAdd(matA, matB, matC);

        barrier();
    }

    // ── 5. Write back the 16x16 accumulated tile to global C ──
    uint out_col = tile_col + (sg_id * 16u);
    coopMatStore(matC, C, tile_row * params.N + out_col, params.N,
                 gl_CooperativeMatrixLayoutRowMajor);
}
