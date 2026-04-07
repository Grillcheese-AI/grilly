#version 450

// Fused MoE Layer Backward: compute dx (gradient w.r.t. input) for all 4 experts in one dispatch.
//
// dx[row, k] = sum_e(router_w[e] * sum_col(grad_out[row, col] * expert_w[e][col, k]))
//            = sum_e(router_w[e] * (grad_out[row,:] @ expert_w[e][:, k]))
//
// This is: dx = sum_e(w_e * grad_out @ W_e)
// Each expert weight matrix W_e is (d_model, d_model), stored row-major as W_e[col, k].
// So grad_out @ W_e means: for output element (row, k), sum over col: grad[row,col] * W[col,k]
//
// Tiled: 16x16, K tiled in blocks of 16.

layout (local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

// Gradient w.r.t. layer output (seq_len, d_model) — this is dx from the next layer
layout(set = 0, binding = 0) readonly buffer GradOut {
    float grad_out[];
};

// Expert weights (same as forward): 4 experts × (d_model, d_model)
layout(set = 0, binding = 1) readonly buffer ExpertWeights {
    float ew[];
};

// Output: gradient w.r.t. layer input (seq_len, d_model)
layout(set = 0, binding = 2) buffer GradInput {
    float grad_in[];
};

layout(push_constant) uniform PushConsts {
    uint seq_len;
    uint d_model;
    uint n_experts;
    float router_w0;
    float router_w1;
    float router_w2;
    float router_w3;
};

shared float tileG[16][17];  // grad_out tile

void main() {
    uint row = gl_WorkGroupID.y * 16 + gl_LocalInvocationID.y;
    uint k_out = gl_WorkGroupID.x * 16 + gl_LocalInvocationID.x;  // output K index
    uint ty = gl_LocalInvocationID.y;
    uint tx = gl_LocalInvocationID.x;

    if (row >= seq_len || k_out >= d_model) return;

    float acc0 = 0.0, acc1 = 0.0, acc2 = 0.0, acc3 = 0.0;

    uint numTiles = (d_model + 15) / 16;
    uint dd = d_model * d_model;
    uint row_base = row * d_model;

    for (uint t = 0; t < numTiles; t++) {
        uint col_base = t * 16;

        // Load grad_out tile
        uint col = col_base + tx;
        tileG[ty][tx] = (col < d_model) ? grad_out[row_base + col] : 0.0;

        barrier();

        // Accumulate: dx[row, k] += grad[row, col] * W[e][col, k]
        // W layout: ew[e * d*d + col * d + k]
        for (uint c = 0; c < 16; c += 4) {
            uint col0 = col_base + c;
            float g0 = tileG[ty][c];
            float g1 = tileG[ty][c + 1];
            float g2 = tileG[ty][c + 2];
            float g3 = tileG[ty][c + 3];

            // Expert 0: W[0][col, k_out]
            acc0 = fma(g0, ew[0 * dd + (col0)     * d_model + k_out], acc0);
            acc0 = fma(g1, ew[0 * dd + (col0 + 1) * d_model + k_out], acc0);
            acc0 = fma(g2, ew[0 * dd + (col0 + 2) * d_model + k_out], acc0);
            acc0 = fma(g3, ew[0 * dd + (col0 + 3) * d_model + k_out], acc0);

            // Expert 1
            acc1 = fma(g0, ew[1 * dd + (col0)     * d_model + k_out], acc1);
            acc1 = fma(g1, ew[1 * dd + (col0 + 1) * d_model + k_out], acc1);
            acc1 = fma(g2, ew[1 * dd + (col0 + 2) * d_model + k_out], acc1);
            acc1 = fma(g3, ew[1 * dd + (col0 + 3) * d_model + k_out], acc1);

            // Expert 2
            acc2 = fma(g0, ew[2 * dd + (col0)     * d_model + k_out], acc2);
            acc2 = fma(g1, ew[2 * dd + (col0 + 1) * d_model + k_out], acc2);
            acc2 = fma(g2, ew[2 * dd + (col0 + 2) * d_model + k_out], acc2);
            acc2 = fma(g3, ew[2 * dd + (col0 + 3) * d_model + k_out], acc2);

            // Expert 3
            acc3 = fma(g0, ew[3 * dd + (col0)     * d_model + k_out], acc3);
            acc3 = fma(g1, ew[3 * dd + (col0 + 1) * d_model + k_out], acc3);
            acc3 = fma(g2, ew[3 * dd + (col0 + 2) * d_model + k_out], acc3);
            acc3 = fma(g3, ew[3 * dd + (col0 + 3) * d_model + k_out], acc3);
        }

        barrier();
    }

    // Weighted sum + residual gradient (dx passes through residual)
    float dx = router_w0 * acc0 + router_w1 * acc1
             + router_w2 * acc2 + router_w3 * acc3;

    // Residual: grad_input = grad_out + dx (grad passes through addition)
    uint idx = row * d_model + k_out;
    grad_in[idx] = grad_out[idx] + dx;
}
