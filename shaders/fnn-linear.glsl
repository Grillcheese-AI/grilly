#version 450

// Tiled GEMM: output = input @ W^T + bias
// 16x16 tiles, shared memory, 4-way unrolled K loop.
// Correct and stable. Performance limited by per-op dispatch overhead.

layout (local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer Input {
    float input_data[];
};

layout(set = 0, binding = 1) readonly buffer Weights {
    float W[];
};

layout(set = 0, binding = 2) readonly buffer Bias {
    float b[];
};

layout(set = 0, binding = 3) buffer Output {
    float output_data[];
};

layout(push_constant) uniform PushConsts {
    uint batch_seq;
    uint input_dim;
    uint output_dim;
    uint has_bias;
};

shared float tileA[16][17];  // +1 padding to avoid bank conflicts
shared float tileB[16][17];

void main() {
    uint row = gl_WorkGroupID.y * 16 + gl_LocalInvocationID.y;
    uint col = gl_WorkGroupID.x * 16 + gl_LocalInvocationID.x;
    uint ty = gl_LocalInvocationID.y;
    uint tx = gl_LocalInvocationID.x;

    float acc = 0.0;
    uint numTiles = (input_dim + 15) / 16;

    for (uint t = 0; t < numTiles; t++) {
        uint k_base = t * 16;

        uint k_a = k_base + tx;
        tileA[ty][tx] = (row < batch_seq && k_a < input_dim)
            ? input_data[row * input_dim + k_a] : 0.0;

        uint k_b = k_base + ty;
        tileB[ty][tx] = (col < output_dim && k_b < input_dim)
            ? W[col * input_dim + k_b] : 0.0;

        barrier();

        for (uint k = 0; k < 16; k += 4) {
            acc += tileA[ty][k]     * tileB[k][tx];
            acc += tileA[ty][k + 1] * tileB[k + 1][tx];
            acc += tileA[ty][k + 2] * tileB[k + 2][tx];
            acc += tileA[ty][k + 3] * tileB[k + 3][tx];
        }

        barrier();
    }

    if (row < batch_seq && col < output_dim) {
        if (has_bias == 1) acc += b[col];
        output_data[row * output_dim + col] = acc;
    }
}
