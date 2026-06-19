#version 450
// Split a fused QKV buffer of shape (B, S, 3*H*Dh) into three separate
// buffers in (B, H, S, Dh) BHSD layout. Used to bridge the QKV linear output
// (BSHD order) with the chunked-sliding-window attention shader (BHSD order).
//
// Layout:
//   input[bs * 3*H*Dh + qkv * H*Dh + h*Dh + d]  (BSHD3Dh packed)
//   Q[b * H*S*Dh + h * S*Dh + s * Dh + d]        (BHSD)
//   K[b * H*S*Dh + h * S*Dh + s * Dh + d]        (BHSD)
//   V[b * H*S*Dh + h * S*Dh + s * Dh + d]        (BHSD)
//
// Push constants: batch_size, num_heads, seq_len, head_dim (16 bytes)
// Workgroup: (64,)
// Dispatch: (ceil(B*H*S*Dh/64), 1, 1)

layout(local_size_x = 64) in;

layout(set=0, binding=0) readonly  buffer Input { float In[]; };
layout(set=0, binding=1) writeonly buffer OutQ   { float Q[]; };
layout(set=0, binding=2) writeonly buffer OutK   { float K[]; };
layout(set=0, binding=3) writeonly buffer OutV   { float V[]; };

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

    // output BHSD index -> (b, h, s, d)
    uint d = gID % head_dim;
    uint s = (gID / head_dim) % seq_len;
    uint h = (gID / (head_dim * seq_len)) % num_heads;
    uint b = gID / (head_dim * seq_len * num_heads);

    uint bs = b * seq_len + s;
    uint hd_d = h * head_dim + d;
    uint stride3 = num_heads * head_dim;

    Q[gID] = In[bs * 3 * stride3 + 0 * stride3 + hd_d];
    K[gID] = In[bs * 3 * stride3 + 1 * stride3 + hd_d];
    V[gID] = In[bs * 3 * stride3 + 2 * stride3 + hd_d];
}
