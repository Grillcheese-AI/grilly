// tensor-where.glsl
// Elementwise conditional select: out[i] = cond[i] > 0 ? a[i] : b[i]
// All three inputs (cond, a, b) and output have `count` elements.
#version 450
layout(local_size_x = 256) in;

layout(set = 0, binding = 0) readonly buffer Cond   { float cond[]; };
layout(set = 0, binding = 1) readonly buffer A      { float a[]; };
layout(set = 0, binding = 2) readonly buffer B      { float b[]; };
layout(set = 0, binding = 3) writeonly buffer Output { float out_data[]; };

layout(push_constant) uniform Params {
    uint count;
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= count) return;
    out_data[idx] = cond[idx] > 0.0 ? a[idx] : b[idx];
}
