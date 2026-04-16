#version 450
#extension GL_EXT_control_flow_attributes : enable

// Fused MoE Layer (vec4): 4 expert matmuls + blend + residual.
// RDNA2 optimized: vec4 loads (128-bit), 16x16 workgroup, LDS bank padding.
//
// output[row, col] = input[row, col] + sum_e(w[e] * (input[row,:] @ expert_w[e][col,:]))
// All buffers use vec4 packing (d_model must be multiple of 4).

layout (local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0) readonly buffer Input { vec4 x[]; };
layout(set = 0, binding = 1) readonly buffer ExpertWeights { vec4 ew[]; };
layout(set = 0, binding = 2) writeonly buffer Output { vec4 out_data[]; };
layout(set = 0, binding = 3) readonly buffer RouterScratch { float rscratch[]; };

layout(push_constant) uniform PushConsts {
    uint seq_len;
    uint d_model;       // must be multiple of 4
    uint n_experts;
    uint weights_offset;
};

shared vec4 tileX[16][17];  // input tile with bank-conflict padding

void main() {
    uint row = gl_WorkGroupID.y * 16 + gl_LocalInvocationID.y;
    uint col_vec = gl_WorkGroupID.x * 16 + gl_LocalInvocationID.x;
    uint tx = gl_LocalInvocationID.x;
    uint ty = gl_LocalInvocationID.y;

    uint d_vec = d_model / 4;

    if (row >= seq_len || col_vec >= d_vec) return;

    // Load router weights
    float w0 = rscratch[weights_offset];
    float w1 = rscratch[weights_offset + 1];
    float w2 = rscratch[weights_offset + 2];
    float w3 = rscratch[weights_offset + 3];

    vec4 acc0 = vec4(0.0), acc1 = vec4(0.0), acc2 = vec4(0.0), acc3 = vec4(0.0);

    uint numTiles = (d_vec + 15) / 16;
    // Expert weight stride: each expert is d_model rows × d_vec vec4s per row
    // ew layout: ew[expert * d_model * d_vec + col * d_vec + k_vec]
    // But we need W[col_out, k_in] which is ew[expert * d * d_vec + col_vec*4*d_vec + k_vec]
    // Simplified: expert weights are (d, d) row-major packed as vec4
    // ew[expert * d * d_vec + row_w * d_vec + k_vec]
    uint ew_stride = d_model * d_vec;  // vec4s per expert

    [[unroll]]
    for (uint t = 0; t < numTiles; t++) {
        uint k_base = t * 16;

        // Load input tile: x[row, k_base*4 .. (k_base+16)*4]
        uint k_vec = k_base + tx;
        tileX[ty][tx] = (k_vec < d_vec) ? x[row * d_vec + k_vec] : vec4(0.0);

        memoryBarrierShared();
        barrier();

        // Accumulate for all 4 experts
        uint k_limit = min(16u, d_vec - k_base);
        for (uint k = 0; k < k_limit; k++) {
            vec4 xv = tileX[ty][k];
            uint k_global = k_base + k;

            // Expert 0
            acc0.x += dot(xv, ew[0 * ew_stride + (col_vec * 4 + 0) * d_vec + k_global]);
            acc0.y += dot(xv, ew[0 * ew_stride + (col_vec * 4 + 1) * d_vec + k_global]);
            acc0.z += dot(xv, ew[0 * ew_stride + (col_vec * 4 + 2) * d_vec + k_global]);
            acc0.w += dot(xv, ew[0 * ew_stride + (col_vec * 4 + 3) * d_vec + k_global]);

            // Expert 1
            acc1.x += dot(xv, ew[1 * ew_stride + (col_vec * 4 + 0) * d_vec + k_global]);
            acc1.y += dot(xv, ew[1 * ew_stride + (col_vec * 4 + 1) * d_vec + k_global]);
            acc1.z += dot(xv, ew[1 * ew_stride + (col_vec * 4 + 2) * d_vec + k_global]);
            acc1.w += dot(xv, ew[1 * ew_stride + (col_vec * 4 + 3) * d_vec + k_global]);

            // Expert 2
            acc2.x += dot(xv, ew[2 * ew_stride + (col_vec * 4 + 0) * d_vec + k_global]);
            acc2.y += dot(xv, ew[2 * ew_stride + (col_vec * 4 + 1) * d_vec + k_global]);
            acc2.z += dot(xv, ew[2 * ew_stride + (col_vec * 4 + 2) * d_vec + k_global]);
            acc2.w += dot(xv, ew[2 * ew_stride + (col_vec * 4 + 3) * d_vec + k_global]);

            // Expert 3
            acc3.x += dot(xv, ew[3 * ew_stride + (col_vec * 4 + 0) * d_vec + k_global]);
            acc3.y += dot(xv, ew[3 * ew_stride + (col_vec * 4 + 1) * d_vec + k_global]);
            acc3.z += dot(xv, ew[3 * ew_stride + (col_vec * 4 + 2) * d_vec + k_global]);
            acc3.w += dot(xv, ew[3 * ew_stride + (col_vec * 4 + 3) * d_vec + k_global]);
        }

        barrier();
    }

    vec4 blended = w0 * acc0 + w1 * acc1 + w2 * acc2 + w3 * acc3;
    uint out_idx = row * d_vec + col_vec;
    out_data[out_idx] = x[out_idx] + blended;
}
