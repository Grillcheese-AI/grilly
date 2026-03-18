// tensor-bmm.glsl
// Batched matrix multiply: out[b] = a[b] @ b_mat[b]
// a    shape: (batch, M, K)
// b_mat shape: (batch, K, N)
// out  shape: (batch, M, N)
// Dispatch: (M, N, batch) threads via local_size (16, 16, 1).
#version 450
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer InputA  { float a[]; };
layout(set = 0, binding = 1) readonly buffer InputB  { float b_mat[]; };
layout(set = 0, binding = 2) writeonly buffer Output { float out_data[]; };

layout(push_constant) uniform Params {
    uint batch;
    uint M;
    uint K;
    uint N;
};

void main() {
    uint row       = gl_GlobalInvocationID.x;
    uint col       = gl_GlobalInvocationID.y;
    uint batch_idx = gl_GlobalInvocationID.z;
    if (row >= M || col >= N || batch_idx >= batch) return;

    float sum    = 0.0;
    uint a_offset = batch_idx * M * K;
    uint b_offset = batch_idx * K * N;
    for (uint k = 0; k < K; k++) {
        sum += a[a_offset + row * K + k] * b_mat[b_offset + k * N + col];
    }
    out_data[batch_idx * M * N + row * N + col] = sum;
}
