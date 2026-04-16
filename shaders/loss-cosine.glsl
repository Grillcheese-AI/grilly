#version 450

layout(local_size_x = 64) in;

layout(binding = 0) readonly  buffer PBuf { float P[]; };
layout(binding = 1) readonly  buffer TBuf { float T[]; };
layout(binding = 2) writeonly buffer LBuf { float L[]; };

layout(push_constant) uniform Push {
    uint batch_size;
    uint dim;
} params;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= params.batch_size) return;
    
    float dp = 0.0;
    float pnorm2 = 0.0;
    float tnorm2 = 0.0;
    
    uint offset = i * params.dim;
    for (uint d = 0; d < params.dim; ++d) {
        float pd = P[offset + d];
        float td = T[offset + d];
        dp += pd * td;
        pnorm2 += pd * pd;
        tnorm2 += td * td;
    }
    
    float pnorm = sqrt(pnorm2) + 1e-9;
    float tnorm = sqrt(tnorm2) + 1e-9;
    
    L[i] = 1.0 - (dp / (pnorm * tnorm));
}
