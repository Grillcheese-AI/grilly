#version 450
layout(local_size_x = 256) in;
layout(std430, set = 0, binding = 0) buffer BufA { float a[]; };
layout(std430, set = 0, binding = 1) readonly buffer BufB { float b[]; };
layout(push_constant) uniform Params { uint total; };
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= total) return;
    a[i] += b[i];
}
