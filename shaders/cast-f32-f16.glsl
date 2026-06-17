#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

// cast-f32-f16.glsl
// Elementwise fp32 -> fp16 copy. Used to stage resident fp32 buffers into the
// fp16 operands that gemm-coopmat-shared requires (A, B). Output is float16_t.
// 1D dispatch over total_elements, 256 threads per workgroup.

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(std430, set = 0, binding = 0) readonly buffer InBuf  { float    data_in[];  };
layout(std430, set = 0, binding = 1) writeonly buffer OutBuf { float16_t data_out[]; };

layout(push_constant) uniform Params {
    uint total_elements;
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= total_elements) return;
    data_out[idx] = float16_t(data_in[idx]);
}
