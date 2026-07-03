#version 450
#extension GL_KHR_shader_subgroup_arithmetic : enable

// Fused sampled-BCE (NCE/SGNS) LM head: loss + dH + dW in ONE submit.
//
// The softmax-free head for cubby: instead of logits = H @ W^T over the full
// vocab (the O(d*V) matmul that dominates the small-trunk budget), each token
// scores ONLY its true target + K sampled negatives:
//
//   s[n,j]    = dot(H[n], W[ids[n,j]])          ids[n,0] = target
//   loss[n]   = softplus(-s[n,0]) + sum_{j>0} softplus(s[n,j])
//   dL/ds     = -sigmoid(-s)*invN  (pos)  /  sigmoid(s)*invN  (neg)
//   dH[n]     = sum_j dL/ds[n,j] * W[ids[n,j]]
//   dW[id]   += dL/ds[n,j] * H[n]
//
// Negatives that collide with their own target (ids[n,j] == ids[n,0], j>0)
// contribute nothing. No softmax, no normalisation over V, no (N,V) tensor.
//
// NO GL_EXT_shader_atomic_float ANYWHERE. Float buffer atomicAdd was measured
// broken on this stack (all accumulations landed at flat index 0 — see
// test_sampled_bce_parity failures of the v1 kernel; convd_col2im_noatomic
// is the same lesson learned earlier). Instead:
//   - losses: workgroup-per-token + subgroup/LDS reduction, ONE plain store
//     (the proven loss-ce-fused idiom).
//   - dW: CAS-loop float add via core atomicCompSwap on a uint-typed view of
//     the buffer — baseline Vulkan semantics, correct on every driver.
//
// Two passes via pass_type (one pipeline, one submit — the ~25ms fixed cost
// is per SUBMIT; see cubby gpu_linear.py notes):
//   pass 0: workgroup per token   — scores, per-token loss (reduced), dscore
//   pass 1: thread per (token,dd) — dH accumulate + dW CAS scatter
//
// dW MUST be zero-filled before pass 1 (CommandBatch::fillZero); losses is
// fully written by pass 0 (no zero-fill needed).

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Hidden states (n_tokens, dim), row-major.
layout(set = 0, binding = 0) readonly buffer Hidden {
    float H[];
};

// Tied embedding table (vocab, dim), row-major.
layout(set = 0, binding = 1) readonly buffer Table {
    float W[];
};

// Candidate ids (n_tokens, n_cand) as uint32; column 0 is the true target.
layout(set = 0, binding = 2) readonly buffer Ids {
    uint ids[];
};

// dL/ds (n_tokens, n_cand) — written pass 0, read pass 1.
layout(set = 0, binding = 3) buffer DScore {
    float dscore[];
};

// Per-token loss (n_tokens,) — one plain store per workgroup in pass 0.
layout(set = 0, binding = 4) writeonly buffer Losses {
    float losses[];
};

// Gradient w.r.t. hidden (n_tokens, dim) — fully written pass 1.
layout(set = 0, binding = 5) writeonly buffer GradH {
    float dH[];
};

// Gradient w.r.t. table (vocab, dim) — viewed as raw uint bits so the CAS
// float-add works with core integer atomics. Same buffer bytes the C++ side
// zero-fills and downloads as float32 (bit pattern of 0u == 0.0f).
layout(set = 0, binding = 6) buffer GradW {
    uint dW_bits[];
};

layout(push_constant) uniform PushConsts {
    uint n_tokens;
    uint n_cand;
    uint dim;
    uint pass_type;
    float inv_n;      // 1/n_tokens — bakes the mean into the grads
};

// One LDS slot per subgroup (256 / 32 = 8 subgroups on Wave32; 16 for
// headroom on 16-wide-subgroup parts). Same sizing as loss-ce-fused.glsl.
shared float lds_loss[16];

// Stable softplus(x) = log(1 + exp(x)) = max(x,0) + log(1 + exp(-|x|)).
float softplus_stable(float x) {
    return max(x, 0.0) + log(1.0 + exp(-abs(x)));
}

float sigmoid_stable(float x) {
    return 1.0 / (1.0 + exp(-clamp(x, -30.0, 30.0)));
}

// Core-Vulkan float atomic add: CAS loop over the uint bit pattern.
void atomic_add_f32(uint idx, float v) {
    uint actual = dW_bits[idx];
    uint expected;
    do {
        expected = actual;
        float updated = uintBitsToFloat(expected) + v;
        actual = atomicCompSwap(dW_bits[idx], expected,
                                floatBitsToUint(updated));
    } while (actual != expected);
}

void main() {
    if (pass_type == 0u) {
        // ── Pass 0: one WORKGROUP per token ──────────────────────────────
        uint n = gl_WorkGroupID.x;
        if (n >= n_tokens) return;               // uniform across the group

        uint tid    = gl_LocalInvocationID.x;
        uint sg_idx = gl_SubgroupID;
        uint n_sgs  = gl_NumSubgroups;

        uint cbase = n * n_cand;
        uint hbase = n * dim;
        uint tgt   = ids[cbase];                 // column 0 = true target

        float local_loss = 0.0;
        for (uint j = tid; j < n_cand; j += 256u) {
            uint id = ids[cbase + j];
            uint wbase = id * dim;
            float s = 0.0;
            for (uint dd = 0u; dd < dim; ++dd) {
                s += H[hbase + dd] * W[wbase + dd];
            }

            float ds;
            if (j == 0u) {
                // positive: -log sigmoid(s) = softplus(-s)
                local_loss += softplus_stable(-s);
                ds = -sigmoid_stable(-s) * inv_n;
            } else if (id == tgt) {
                // negative collided with its own target: masked out
                ds = 0.0;
            } else {
                // negative: -log sigmoid(-s) = softplus(s)
                local_loss += softplus_stable(s);
                ds = sigmoid_stable(s) * inv_n;
            }
            dscore[cbase + j] = ds;
        }

        // Subgroup reduce, then cross-subgroup combine via LDS (loss-ce-fused
        // idiom). Every thread participates — no early returns above this.
        float sg_loss = subgroupAdd(local_loss);
        if (subgroupElect()) {
            lds_loss[sg_idx] = sg_loss;
        }
        barrier();
        if (tid == 0u) {
            float total = 0.0;
            for (uint s = 0u; s < n_sgs; ++s) {
                total += lds_loss[s];
            }
            losses[n] = total;
        }
    } else {
        // ── Pass 1: one thread per (token, dim element) ──────────────────
        uint gid = gl_GlobalInvocationID.x;
        if (gid >= n_tokens * dim) return;
        uint n = gid / dim;
        uint dd = gid % dim;

        float hval = H[gid];
        float acc = 0.0;
        uint cbase = n * n_cand;
        for (uint j = 0u; j < n_cand; ++j) {
            float ds = dscore[cbase + j];
            if (ds == 0.0) continue;             // masked collision
            uint widx = ids[cbase + j] * dim + dd;
            acc += ds * W[widx];
            atomic_add_f32(widx, ds * hval);
        }
        dH[gid] = acc;
    }
}
