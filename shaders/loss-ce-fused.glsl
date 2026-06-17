#version 450
#extension GL_KHR_shader_subgroup_arithmetic : enable

// Fused cross-entropy loss + gradient (softmax-bypass vocab head).
//
// ONE dispatch computes BOTH per-row CE loss AND grad_logits, sharing a single
// subgroup-reduced max + sum_exp pass per row. This fuses what were two
// separate kernels (loss-cross-entropy.glsl's 3-pass loss + the older
// shared-memory-tree cross-entropy-backward.glsl) into one, keeping the row
// reduction on-chip (RDNA2 Wave32 subgroups) and never materializing a full
// softmax distribution to VRAM.
//
//   loss[row] = lse - logits[row, target]        ; lse = max + log(sum_exp)
//   grad[row, i] = exp(logits[i] - max) / sum_exp - (i == target ? 1 : 0)
//                = softmax(logits)[i] - one_hot(target)[i]
//
// One workgroup per row, 256 threads (8 subgroups of 32 on Wave32). Each
// thread strides over the vocab so arbitrary num_classes is supported (not
// capped at 256). Targets are a uint32 buffer (the C++ op uploads the raw
// uint32 class indices; we read them directly as uint — NOT as float bits).

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Logits (batch_size, num_classes), row-major.
layout(set = 0, binding = 0) readonly buffer Logits {
    float logits[];
};

// Target class indices (batch_size,), as uint32.
layout(set = 0, binding = 1) readonly buffer Targets {
    uint targets[];
};

// Per-row loss (batch_size,).
layout(set = 0, binding = 2) writeonly buffer Losses {
    float losses[];
};

// Gradient w.r.t. logits (batch_size, num_classes).
layout(set = 0, binding = 3) writeonly buffer GradLogits {
    float grad_logits[];
};

layout(push_constant) uniform PushConsts {
    uint batch_size;
    uint num_classes;
};

// One LDS slot per subgroup (256 / 32 = 8 subgroups max on Wave32; sized 16
// for headroom on 16-wide-subgroup parts).
shared float lds_max[16];
shared float lds_sum[16];

void main() {
    uint row = gl_WorkGroupID.x;
    if (row >= batch_size) return;

    uint tid    = gl_LocalInvocationID.x;
    uint sg_id  = gl_SubgroupInvocationID;
    uint sg_idx = gl_SubgroupID;
    uint n_sgs  = gl_NumSubgroups;

    uint base = row * num_classes;

    // ── Pass 1: per-thread partial max over a strided slice of the row ──
    float local_max = -1e30;
    for (uint i = tid; i < num_classes; i += 256u) {
        local_max = max(local_max, logits[base + i]);
    }
    // Subgroup max, then cross-subgroup combine via LDS.
    float sg_max = subgroupMax(local_max);
    if (subgroupElect()) {
        lds_max[sg_idx] = sg_max;
    }
    barrier();

    float row_max = -1e30;
    for (uint s = 0u; s < n_sgs; ++s) {
        row_max = max(row_max, lds_max[s]);
    }

    // ── Pass 2: per-thread partial sum of exp(logit - row_max) ──
    float local_sum = 0.0;
    for (uint i = tid; i < num_classes; i += 256u) {
        // Clamp shifted logit to match the numpy/CPU oracle (clip to [-60, 60]).
        float shifted = clamp(logits[base + i] - row_max, -60.0, 60.0);
        local_sum += exp(shifted);
    }
    float sg_sum = subgroupAdd(local_sum);
    if (subgroupElect()) {
        lds_sum[sg_idx] = sg_sum;
    }
    barrier();

    float sum_exp = 0.0;
    for (uint s = 0u; s < n_sgs; ++s) {
        sum_exp += lds_sum[s];
    }
    sum_exp = max(sum_exp, 1e-12);
    float inv_sum = 1.0 / sum_exp;

    uint target = targets[row];

    // ── Pass 3: write grad row (softmax - one_hot) ──
    for (uint i = tid; i < num_classes; i += 256u) {
        float shifted = clamp(logits[base + i] - row_max, -60.0, 60.0);
        float sm = exp(shifted) * inv_sum;
        float g = sm;
        if (i == target) {
            g -= 1.0;
        }
        grad_logits[base + i] = g;
    }

    // ── Per-row loss: lse - target_logit (one thread does the scalar write) ──
    if (tid == 0u) {
        float lse = row_max + log(sum_exp);
        float target_logit = 0.0;
        if (target < num_classes) {
            target_logit = logits[base + target];
        }
        losses[row] = lse - target_logit;
    }
}
