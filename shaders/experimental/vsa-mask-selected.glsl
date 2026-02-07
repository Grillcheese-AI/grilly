#version 450
// Mask previously selected indices by writing -INF into the score matrix.
// Each invocation handles one row.
//
// bindings:
// 0: scores (float) length rows*stride
// 1: selected_indices (uint) length rows*k
//
// push constants:
//   uint rows
//   uint stride
//   uint in_offset (rank * rows)

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer Scores {
    float scores[];
};

layout(set = 0, binding = 1) readonly buffer SelIdx {
    uint sel_idx[];
};

layout(push_constant) uniform PushConsts {
    uint rows;
    uint stride;
    uint in_offset;
} pc;

void main() {
    uint row = gl_GlobalInvocationID.x;
    if (row >= pc.rows) { return; }

    uint idx = sel_idx[pc.in_offset + row];
    if (idx == uint(-1)) { return; }

    // NOTE: caller guarantees idx < cols
    scores[row * pc.stride + idx] = -3.402823466e38;
}
