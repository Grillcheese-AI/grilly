#version 450
// Merge 3 separate BHSD buffers (dQ, dK, dV) into one (BS, 3*d) BSHD buffer.
// Inverse of attention-qkv-split.glsl.
//
// Input:  Q[b*H*S*Dh + h*S*Dh + s*Dh + d]  (BHSD)
// Output: Out[bs * 3*H*Dh + slot*H*Dh + h*Dh + d]  where bs = b*S+s, slot=0,1,2

layout(local_size_x = 64) in;

layout(set=0, binding=0) readonly buffer QBuf { float qData[]; };
layout(set=0, binding=1) readonly buffer KBuf { float kData[]; };
layout(set=0, binding=2) readonly buffer VBuf { float vData[]; };
layout(set=0, binding=3) writeonly buffer OutBuf { float outData[]; };

layout(push_constant) uniform Params {
    uint batch_size;
    uint num_heads;
    uint seq_len;
    uint head_dim;
};

void main() {
    uint gID = gl_GlobalInvocationID.x;
    uint total = batch_size * num_heads * seq_len * head_dim;
    if (gID >= total) return;

    uint d = gID % head_dim;
    uint s = (gID / head_dim) % seq_len;
    uint h = (gID / (head_dim * seq_len)) % num_heads;
    uint b = gID / (head_dim * seq_len * num_heads);

    uint bs = b * seq_len + s;
    uint hd_d = h * head_dim + d;
    uint stride3 = num_heads * head_dim;

    outData[bs * 3 * stride3 + 0 * stride3 + hd_d] = qData[gID];
    outData[bs * 3 * stride3 + 1 * stride3 + hd_d] = kData[gID];
    outData[bs * 3 * stride3 + 2 * stride3 + hd_d] = vData[gID];
}
