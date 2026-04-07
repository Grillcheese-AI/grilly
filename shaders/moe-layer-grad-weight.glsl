#version 450

// Fused MoE grad_W: compute weight gradients for all 4 experts in one dispatch.
//
// grad_W[e][col, k] = router_w[e] * sum_row(grad_out[row, col] * input[row, k])
//                   = router_w[e] * (grad_out[:,col].T @ input[:,k])
//
// Each thread computes one (col, k) element for all 4 experts.
// Tiled: 16x16, reduction over seq_len in blocks of 16.

layout (local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer GradOut {
    float grad_out[];    // (seq_len, d_model)
};

layout(set = 0, binding = 1) readonly buffer Input {
    float x[];           // (seq_len, d_model)
};

// Output: 4 expert weight gradients packed contiguously
// grad_ew[e * d*d + col * d + k]
layout(set = 0, binding = 2) buffer GradWeights {
    float grad_ew[];     // (4 * d_model * d_model)
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

shared float tileG[16][17];  // grad_out[:, col] tile
shared float tileX[16][17];  // input[:, k] tile

void main() {
    uint col = gl_WorkGroupID.y * 16 + gl_LocalInvocationID.y;
    uint k = gl_WorkGroupID.x * 16 + gl_LocalInvocationID.x;
    uint ty = gl_LocalInvocationID.y;
    uint tx = gl_LocalInvocationID.x;

    if (col >= d_model || k >= d_model) return;

    float acc = 0.0;  // shared across experts (same outer product, different scaling)

    uint numTiles = (seq_len + 15) / 16;

    for (uint t = 0; t < numTiles; t++) {
        uint row_base = t * 16;

        // Load grad_out column tile: grad_out[row_base+ty, col]
        uint r_g = row_base + ty;
        tileG[ty][tx] = (r_g < seq_len && col < d_model)
            ? grad_out[r_g * d_model + col] : 0.0;

        // Load input column tile: x[row_base+ty, k]
        uint r_x = row_base + ty;
        tileX[ty][tx] = (r_x < seq_len && k < d_model)
            ? x[r_x * d_model + k] : 0.0;

        barrier();

        // Accumulate outer product: sum_row(grad[row, col] * x[row, k])
        for (uint r = 0; r < 16; r += 4) {
            acc = fma(tileG[r][ty], tileX[r][tx], acc);      // Note: ty indexes col, tx indexes k
            acc = fma(tileG[r+1][ty], tileX[r+1][tx], acc);
            acc = fma(tileG[r+2][ty], tileX[r+2][tx], acc);
            acc = fma(tileG[r+3][ty], tileX[r+3][tx], acc);
        }

        barrier();
    }

    // Write grad_W for all 4 experts (same outer product, scaled by router weight)
    uint dd = d_model * d_model;
    uint base = col * d_model + k;
    grad_ew[0 * dd + base] = router_w0 * acc;
    grad_ew[1 * dd + base] = router_w1 * acc;
    grad_ew[2 * dd + base] = router_w2 * acc;
    grad_ew[3 * dd + base] = router_w3 * acc;
}
