#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

// transpose-f32-f16.glsl
// Fused transpose + cast: read fp32 in[r][c], write fp16 out[c][r].
// Mirrors tensor-transpose.glsl but the output is float16_t and the value is
// cast on store. Used to build g^T (fp16) for the grad_weight coopmat GEMM
// without a separate transpose+cast round trip.
// Push constants: rows, cols (of the INPUT matrix).
// 1D dispatch over rows*cols, 256 threads per workgroup.

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(std430, set = 0, binding = 0) readonly buffer InBuf  { float    data_in[];  };
layout(std430, set = 0, binding = 1) writeonly buffer OutBuf { float16_t data_out[]; };

layout(push_constant) uniform Params {
    uint rows;
    uint cols;
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= rows * cols) return;
    uint r = idx / cols;
    uint c = idx % cols;
    data_out[c * rows + r] = float16_t(data_in[r * cols + c]);
}
