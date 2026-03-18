// tensor-transpose.glsl
// Transpose a 2D matrix: out[c][r] = in[r][c]
// Push constants: rows, cols (of the INPUT matrix)
#version 450
layout(local_size_x = 256) in;

layout(set = 0, binding = 0) readonly buffer Input  { float data_in[]; };
layout(set = 0, binding = 1) writeonly buffer Output { float data_out[]; };

layout(push_constant) uniform Params {
    uint rows;
    uint cols;
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= rows * cols) return;
    uint r = idx / cols;
    uint c = idx % cols;
    data_out[c * rows + r] = data_in[r * cols + c];
}
