// tensor-cat.glsl
// Concatenate two tensors along their last dimension.
// batch_size = product of all dims except the last.
// dim_a = last-dim width of tensor A.
// dim_b = last-dim width of tensor B.
// Output shape: (batch_size, dim_a + dim_b)
#version 450
layout(local_size_x = 256) in;

layout(set = 0, binding = 0) readonly buffer InputA  { float a[]; };
layout(set = 0, binding = 1) readonly buffer InputB  { float b[]; };
layout(set = 0, binding = 2) writeonly buffer Output { float out_data[]; };

layout(push_constant) uniform Params {
    uint batch_size;
    uint dim_a;
    uint dim_b;
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint total_out = batch_size * (dim_a + dim_b);
    if (idx >= total_out) return;
    uint batch_idx = idx / (dim_a + dim_b);
    uint inner_idx = idx % (dim_a + dim_b);
    if (inner_idx < dim_a) {
        out_data[idx] = a[batch_idx * dim_a + inner_idx];
    } else {
        out_data[idx] = b[batch_idx * dim_b + (inner_idx - dim_a)];
    }
}
