#version 450

// Fused MoE Layer: 4 expert matmuls + weighted blend + residual in ONE dispatch.
//
// Router weights read from scratch buffer (computed by moe-router shader).
// No CPU round-trip between layers.
//
// output[row, col] = input[row, col] + sum_e(w[e] * (input[row,:] @ expert_w[e][col,:]))

layout (local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer Input {
    float x[];
};

layout(set = 0, binding = 1) readonly buffer ExpertWeights {
    float ew[];   // 4 experts packed: ew[e * d*d + col * d + k]
};

layout(set = 0, binding = 2) buffer Output {
    float out_data[];
};

// Router scratch buffer: [x_mean (d) | logits (E) | max (1) | sum (1) | weights (E)]
layout(set = 0, binding = 3) readonly buffer RouterScratch {
    float rscratch[];
};

layout(push_constant) uniform PushConsts {
    uint seq_len;
    uint d_model;
    uint n_experts;
    uint weights_offset;  // offset in rscratch where softmax weights start
};

shared float tileX[16][17];

void main() {
    uint row = gl_WorkGroupID.y * 16 + gl_LocalInvocationID.y;
    uint col = gl_WorkGroupID.x * 16 + gl_LocalInvocationID.x;
    uint ty = gl_LocalInvocationID.y;
    uint tx = gl_LocalInvocationID.x;

    if (row >= seq_len || col >= d_model) return;

    // Read router weights from scratch buffer
    float w0 = rscratch[weights_offset];
    float w1 = rscratch[weights_offset + 1];
    float w2 = rscratch[weights_offset + 2];
    float w3 = rscratch[weights_offset + 3];

    float acc0 = 0.0, acc1 = 0.0, acc2 = 0.0, acc3 = 0.0;

    uint numTiles = (d_model + 15) / 16;
    uint row_base = row * d_model;
    uint dd = d_model * d_model;

    for (uint t = 0; t < numTiles; t++) {
        uint k_base = t * 16;

        uint k_a = k_base + tx;
        tileX[ty][tx] = (k_a < d_model) ? x[row_base + k_a] : 0.0;

        barrier();

        uint e0_base = 0 * dd + col * d_model + k_base;
        uint e1_base = 1 * dd + col * d_model + k_base;
        uint e2_base = 2 * dd + col * d_model + k_base;
        uint e3_base = 3 * dd + col * d_model + k_base;

        for (uint k = 0; k < 16; k += 4) {
            float x0 = tileX[ty][k];
            float x1 = tileX[ty][k + 1];
            float x2 = tileX[ty][k + 2];
            float x3 = tileX[ty][k + 3];

            acc0 = fma(x0, ew[e0_base + k],     acc0);
            acc0 = fma(x1, ew[e0_base + k + 1], acc0);
            acc0 = fma(x2, ew[e0_base + k + 2], acc0);
            acc0 = fma(x3, ew[e0_base + k + 3], acc0);

            acc1 = fma(x0, ew[e1_base + k],     acc1);
            acc1 = fma(x1, ew[e1_base + k + 1], acc1);
            acc1 = fma(x2, ew[e1_base + k + 2], acc1);
            acc1 = fma(x3, ew[e1_base + k + 3], acc1);

            acc2 = fma(x0, ew[e2_base + k],     acc2);
            acc2 = fma(x1, ew[e2_base + k + 1], acc2);
            acc2 = fma(x2, ew[e2_base + k + 2], acc2);
            acc2 = fma(x3, ew[e2_base + k + 3], acc2);

            acc3 = fma(x0, ew[e3_base + k],     acc3);
            acc3 = fma(x1, ew[e3_base + k + 1], acc3);
            acc3 = fma(x2, ew[e3_base + k + 2], acc3);
            acc3 = fma(x3, ew[e3_base + k + 3], acc3);
        }

        barrier();
    }

    float blended = w0 * acc0 + w1 * acc1 + w2 * acc2 + w3 * acc3;
    uint idx = row * d_model + col;
    out_data[idx] = x[idx] + blended;
}
