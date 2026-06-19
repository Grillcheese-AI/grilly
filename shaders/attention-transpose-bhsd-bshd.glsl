#version 450
// Transpose BHSD (batch, head, seq, dim) -> BSHD (batch, seq, head, dim).
// Used to reshape the attention output from the BHSD layout of the chunked-
// sliding-window attention shader into the (B*S, D) layout consumed by the 
// output projection linear.
//
// Input:  In[b*H*S*Dh + h*S*Dh + s*Dh + d]
// Output: Out[b*S*H*Dh + s*H*Dh + h*Dh + d]     (= (B*S, D) flat)
//
// Push constants: batch_size, num_heads, seq_len, head_dim (16 bytes)

layout(local_size_x = 64) in;

layout(set=0, binding=0) readonly  buffer Input  { float In[]; };
layout(set=0, binding=1) writeonly buffer Output { float Out[]; };

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

    // output BSHD index -> (b, s, h, d)
    uint d = gID % head_dim;
    uint h = (gID / head_dim) % num_heads;
    uint s = (gID / (head_dim * num_heads)) % seq_len;
    uint b = gID / (head_dim * num_heads * seq_len);

    // input BHSD index
    uint in_idx = ((b * num_heads + h) * seq_len + s) * head_dim + d;
    Out[gID] = In[in_idx];
}
