#version 450
// Embedding backward (scatter-add): for each token t with row id = ids[t],
//   E_grad[ids[t], :] += emb_grad[t, :]   accumulating IN PLACE so the tied-head
// weight grad already in E_grad and the embedding-table grad merge on-GPU.
//
// Float add via atomicCompSwap on the uint bit-pattern -- NOT atomicAdd(float):
// the latter (GL_EXT_shader_atomic_float / OpAtomicFAddEXT) is wave-coalesced
// incorrectly on RDNA when lanes scatter to different addresses (observed: a
// single +1 became +32 = wavefront size). CAS is a true per-lane read-modify-
// write and is correct. One thread per (token, dim) element.

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer EmbGrad { float g[]; };    // tokens*dim
layout(set = 0, binding = 1) readonly buffer Ids     { uint  ids[]; };  // tokens (u32)
layout(set = 0, binding = 2)          buffer EGrad   { uint  eg[]; };   // vocab*dim (float bits)

layout(push_constant) uniform PushConsts { uint tokens; uint dim; };

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= tokens * dim) return;
    uint tok = idx / dim;
    uint j   = idx - tok * dim;
    uint k   = ids[tok] * dim + j;
    float add = g[idx];

    uint old = eg[k];
    uint assumed;
    do {
        assumed = old;
        uint newbits = floatBitsToUint(uintBitsToFloat(assumed) + add);
        old = atomicCompSwap(eg[k], assumed, newbits);
    } while (old != assumed);
}
