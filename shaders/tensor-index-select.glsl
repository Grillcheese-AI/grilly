// tensor-index-select.glsl
// Gather rows from a 2D input by integer indices.
// out[i, :] = input[indices[i], :]
// num_indices = number of output rows; row_size = width of each row.
#version 450
layout(local_size_x = 256) in;

layout(set = 0, binding = 0) readonly buffer Input   { float data_in[]; };
layout(set = 0, binding = 1) readonly buffer Indices { uint  indices[]; };
layout(set = 0, binding = 2) writeonly buffer Output  { float data_out[]; };

layout(push_constant) uniform Params {
    uint num_indices;
    uint row_size;
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= num_indices * row_size) return;
    uint row     = idx / row_size;
    uint col     = idx % row_size;
    uint src_row = indices[row];
    data_out[idx] = data_in[src_row * row_size + col];
}
