#version 450

// Fused MoE Backward (vec4): grad_input for all 4 experts.
// RDNA2 optimized: vec4 loads, 32x8 workgroup (matches Wave32), LDS padding.
//
// dx[row, k] = grad_out[row, k] + sum_e(w_e * sum_col(grad_out[row, col] * W_e[col, k]))
// All buffers vec4 packed (d_model must be multiple of 4).

layout (local_size_x = 32, local_size_y = 8) in;

layout(set = 0, binding = 0) readonly buffer GradOut { vec4 g_out[]; };
layout(set = 0, binding = 1) readonly buffer ExpertWeights { vec4 ew[]; };
layout(set = 0, binding = 2) writeonly buffer GradInput { vec4 g_in[]; };

layout(push_constant) uniform PushConsts {
    uint seq_len;
    uint d_model;
    uint n_experts;
    float w0, w1, w2, w3;
};

shared vec4 tileG[8][33];  // grad_out tile, 8 rows × 32 vec4 cols + padding

void main() {
    uint row = gl_WorkGroupID.y * 8 + gl_LocalInvocationID.y;
    uint k_vec = gl_WorkGroupID.x * 32 + gl_LocalInvocationID.x;
    uint ty = gl_LocalInvocationID.y;
    uint tx = gl_LocalInvocationID.x;

    uint d_vec = d_model / 4;

    if (row >= seq_len || k_vec >= d_vec) return;

    vec4 acc0 = vec4(0.0), acc1 = vec4(0.0), acc2 = vec4(0.0), acc3 = vec4(0.0);

    uint numTiles = (d_vec + 31) / 32;
    uint ew_stride = d_model * d_vec;  // vec4s per expert

    for (uint t = 0; t < numTiles; t++) {
        // Load grad_out into LDS
        uint col_vec = t * 32 + tx;
        tileG[ty][tx] = (col_vec < d_vec) ? g_out[row * d_vec + col_vec] : vec4(0.0);

        memoryBarrierShared();
        barrier();

        // Accumulate: dx[k] += grad[col] * W[col, k]
        for (uint c = 0; c < 32; c++) {
            vec4 g = tileG[ty][c];
            uint col_global = t * 32 + c;  // vec4 index
            if (col_global >= d_vec) break;

            // Each vec4 g has 4 scalar grad components for 4 adjacent columns
            // W[col, k] accessed as ew[expert * stride + col_scalar * d_vec + k_vec]

            // Expert 0
            acc0 += g.x * ew[0 * ew_stride + (col_global * 4 + 0) * d_vec + k_vec];
            acc0 += g.y * ew[0 * ew_stride + (col_global * 4 + 1) * d_vec + k_vec];
            acc0 += g.z * ew[0 * ew_stride + (col_global * 4 + 2) * d_vec + k_vec];
            acc0 += g.w * ew[0 * ew_stride + (col_global * 4 + 3) * d_vec + k_vec];

            // Expert 1
            acc1 += g.x * ew[1 * ew_stride + (col_global * 4 + 0) * d_vec + k_vec];
            acc1 += g.y * ew[1 * ew_stride + (col_global * 4 + 1) * d_vec + k_vec];
            acc1 += g.z * ew[1 * ew_stride + (col_global * 4 + 2) * d_vec + k_vec];
            acc1 += g.w * ew[1 * ew_stride + (col_global * 4 + 3) * d_vec + k_vec];

            // Expert 2
            acc2 += g.x * ew[2 * ew_stride + (col_global * 4 + 0) * d_vec + k_vec];
            acc2 += g.y * ew[2 * ew_stride + (col_global * 4 + 1) * d_vec + k_vec];
            acc2 += g.z * ew[2 * ew_stride + (col_global * 4 + 2) * d_vec + k_vec];
            acc2 += g.w * ew[2 * ew_stride + (col_global * 4 + 3) * d_vec + k_vec];

            // Expert 3
            acc3 += g.x * ew[3 * ew_stride + (col_global * 4 + 0) * d_vec + k_vec];
            acc3 += g.y * ew[3 * ew_stride + (col_global * 4 + 1) * d_vec + k_vec];
            acc3 += g.z * ew[3 * ew_stride + (col_global * 4 + 2) * d_vec + k_vec];
            acc3 += g.w * ew[3 * ew_stride + (col_global * 4 + 3) * d_vec + k_vec];
        }

        barrier();
    }

    vec4 dx = w0 * acc0 + w1 * acc1 + w2 * acc2 + w3 * acc3;
    uint idx = row * d_vec + k_vec;
    g_in[idx] = g_out[idx] + dx;  // residual gradient
}
