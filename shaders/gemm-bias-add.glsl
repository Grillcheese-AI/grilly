#version 450

/*
 * Row-broadcast bias add: C[r, c] += bias[c]
 *
 * Used as a second dispatch after gemm-coopmat-shared (which produces C
 * without bias, since coopMatStore can't easily interleave an elementwise
 * add with the tile-cooperative store). 1D dispatch over M*N elements,
 * 256 threads per workgroup.
 */

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0) buffer CBuffer { float C[]; };
layout(binding = 1) readonly buffer BiasBuffer { float bias[]; };

layout(push_constant) uniform PushConstants {
    uint totalElements;  // M * N
    uint N;              // stride (columns per row)
} params;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= params.totalElements) return;
    uint col = idx % params.N;
    C[idx] = C[idx] + bias[col];
}
