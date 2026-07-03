#version 450
#extension GL_KHR_shader_subgroup_arithmetic : enable

// Fused NCE LM head (corrected sampled-BCE): loss + dH + dW + db in ONE submit.
//
// The softmax-free head for cubby. v2 was plain SGNS (word2vec) — it trained
// h.e_target up and h.e_neg down, but never calibrated the ~63k tail tokens
// that unigram^0.75 rarely samples, so full-softmax eval PPL sat near vocab
// size. This is the PROPER NCE objective (Gutmann-Hyvarinen; Mnih-Teh for LMs):
// each score carries the noise-distribution correction, so the trained
// h.e_w + b_w converges to log P(w|ctx) + const and eval softmax reads a real
// distribution.
//
//   G[n,j] = dot(H[n], W[id]) + bias[id] - logkq[id]      logkq = log(k*q_id)
//   loss   = softplus(-G[n,0]) + sum_{j>0} softplus(G[n,j])
//   dL/dG  = -sigmoid(-G)*invN  (pos)  /  sigmoid(G)*invN  (neg)
//   dH[n]  = sum_j dL/dG[n,j] * W[id]
//   dW[id]+= dL/dG[n,j] * H[n]         ; db[id] += dL/dG[n,j]
//
// bias is a learned per-token scalar (init to +logkq so G starts == raw dot).
// use_correction=0 disables both correction+bias (recovers the v2 SGNS head
// for A/B). Target-collision negatives contribute nothing. No softmax, no
// (N,V) tensor, ever.
//
// NO GL_EXT_shader_atomic_float (measured broken on this stack). Losses use
// the loss-ce-fused subgroup/LDS reduction (plain store); dW and db use a
// CAS-loop float add on core uint atomics. dW and db MUST be zero-filled
// before pass 1 (CommandBatch::fillZero).
//
// Two passes via pass_type, one submit:
//   pass 0: workgroup per token   — scores, per-token loss (reduced), dscore
//   pass 1: thread per (token,dd) — dH accumulate + dW CAS scatter
//                                   (thread dd==0 also does the db scatter)

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer Hidden {
    float H[];
};

layout(set = 0, binding = 1) readonly buffer Table {
    float W[];
};

layout(set = 0, binding = 2) readonly buffer Ids {
    uint ids[];
};

// dL/dG (n_tokens, n_cand) — written pass 0, read pass 1.
layout(set = 0, binding = 3) buffer DScore {
    float dscore[];
};

layout(set = 0, binding = 4) writeonly buffer Losses {
    float losses[];
};

layout(set = 0, binding = 5) writeonly buffer GradH {
    float dH[];
};

// Gradient w.r.t. table (vocab, dim) — uint-bits CAS view.
layout(set = 0, binding = 6) buffer GradW {
    uint dW_bits[];
};

// Per-token log(k * q_id) noise correction (vocab,). Read-only.
layout(set = 0, binding = 7) readonly buffer LogKQ {
    float logkq[];
};

// Learned per-token bias (vocab,). Read-only here; its grad goes to GradB.
layout(set = 0, binding = 8) readonly buffer Bias {
    float bias[];
};

// Gradient w.r.t. bias (vocab,) — uint-bits CAS view.
layout(set = 0, binding = 9) buffer GradB {
    uint dB_bits[];
};

layout(push_constant) uniform PushConsts {
    uint n_tokens;
    uint n_cand;
    uint dim;
    uint pass_type;
    float inv_n;
    uint use_correction;   // 1 = add bias - logkq (NCE); 0 = raw dot (SGNS)
};

shared float lds_loss[16];

float softplus_stable(float x) {
    return max(x, 0.0) + log(1.0 + exp(-abs(x)));
}

float sigmoid_stable(float x) {
    return 1.0 / (1.0 + exp(-clamp(x, -30.0, 30.0)));
}

void atomic_add_f32_W(uint idx, float v) {
    uint actual = dW_bits[idx];
    uint expected;
    do {
        expected = actual;
        float updated = uintBitsToFloat(expected) + v;
        actual = atomicCompSwap(dW_bits[idx], expected, floatBitsToUint(updated));
    } while (actual != expected);
}

void atomic_add_f32_B(uint idx, float v) {
    uint actual = dB_bits[idx];
    uint expected;
    do {
        expected = actual;
        float updated = uintBitsToFloat(expected) + v;
        actual = atomicCompSwap(dB_bits[idx], expected, floatBitsToUint(updated));
    } while (actual != expected);
}

void main() {
    if (pass_type == 0u) {
        // ── Pass 0: one WORKGROUP per token ──────────────────────────────
        uint n = gl_WorkGroupID.x;
        if (n >= n_tokens) return;

        uint tid    = gl_LocalInvocationID.x;
        uint sg_idx = gl_SubgroupID;
        uint n_sgs  = gl_NumSubgroups;

        uint cbase = n * n_cand;
        uint hbase = n * dim;
        uint tgt   = ids[cbase];

        float local_loss = 0.0;
        for (uint j = tid; j < n_cand; j += 256u) {
            uint id = ids[cbase + j];
            uint wbase = id * dim;
            float s = 0.0;
            for (uint dd = 0u; dd < dim; ++dd) {
                s += H[hbase + dd] * W[wbase + dd];
            }
            // NCE correction: G = dot + bias[id] - log(k*q_id).
            if (use_correction == 1u) {
                s += bias[id] - logkq[id];
            }

            float ds;
            if (j == 0u) {
                local_loss += softplus_stable(-s);
                ds = -sigmoid_stable(-s) * inv_n;
            } else if (id == tgt) {
                ds = 0.0;
            } else {
                local_loss += softplus_stable(s);
                ds = sigmoid_stable(s) * inv_n;
            }
            dscore[cbase + j] = ds;
        }

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
            if (ds == 0.0) continue;
            uint id = ids[cbase + j];
            uint widx = id * dim + dd;
            acc += ds * W[widx];
            atomic_add_f32_W(widx, ds * hval);
            // db[id] += ds  — dot's bias-derivative is 1; do it once per (n,j)
            // via the dd==0 thread only (bias is per-token, not per-dim).
            if (dd == 0u && use_correction == 1u) {
                atomic_add_f32_B(id, ds);
            }
        }
        dH[gid] = acc;
    }
}
