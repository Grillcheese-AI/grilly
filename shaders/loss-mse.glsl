#version 450

layout(local_size_x = 256) in;

layout(binding = 0) readonly  buffer PBuf { float P[]; };
layout(binding = 1) readonly  buffer TBuf { float T[]; };
layout(binding = 2) writeonly buffer LBuf { float L[]; };

layout(push_constant) uniform Push {
    uint n;
} params;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= params.n) return;
    
    float d = P[i] - T[i];
    L[i] = d * d;
}
