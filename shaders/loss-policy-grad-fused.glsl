#version 450
#extension GL_KHR_shader_subgroup_arithmetic : enable

// Fused policy-gradient head op (GRPO / RLVR). Per row (token position):
//
//   grad[row, i] = coef[row] * (softmax(logits[row])[i] - one_hot(target[row])[i])
//
// coef[row] folds the per-sample advantage AND the completion mask: a masked
// position (prompt / pad / last) passes coef == 0 and gets a zero grad row.
// This is the SFT fused-CE gradient (softmax - one_hot) scaled per row by the
// advantage -- computed entirely on-GPU, so GRPO needs NO host logits readback
// (the (K*S, V) readback that crashed the K=6 step). targets are uint32; the
// ignore-index sentinel target >= num_classes also zeros the row.
//
// One workgroup per row, 256 threads (Wave32 subgroups), strided over the vocab.

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer Logits     { float logits[]; };
layout(set = 0, binding = 1) readonly buffer Targets    { uint  targets[]; };
layout(set = 0, binding = 2) readonly buffer Coef       { float coef[]; };
layout(set = 0, binding = 3) writeonly buffer GradLogits { float grad_logits[]; };

layout(push_constant) uniform PushConsts {
    uint batch_size;
    uint num_classes;
};

shared float lds_max[16];
shared float lds_sum[16];

void main() {
    uint row = gl_WorkGroupID.x;
    if (row >= batch_size) return;

    uint tid    = gl_LocalInvocationID.x;
    uint sg_idx = gl_SubgroupID;
    uint n_sgs  = gl_NumSubgroups;

    uint base   = row * num_classes;
    float c     = coef[row];
    uint target = targets[row];

    // Masked row (coef 0) or ignore-index target -> zero grad slice, skip.
    if (c == 0.0 || target >= num_classes) {
        for (uint i = tid; i < num_classes; i += 256u) grad_logits[base + i] = 0.0;
        return;
    }

    // Pass 1: row max (subgroup + LDS combine).
    float local_max = -1e30;
    for (uint i = tid; i < num_classes; i += 256u) {
        local_max = max(local_max, logits[base + i]);
    }
    float sg_max = subgroupMax(local_max);
    if (subgroupElect()) lds_max[sg_idx] = sg_max;
    barrier();
    float row_max = -1e30;
    for (uint s = 0u; s < n_sgs; ++s) row_max = max(row_max, lds_max[s]);

    // Pass 2: sum exp(logit - row_max).
    float local_sum = 0.0;
    for (uint i = tid; i < num_classes; i += 256u) {
        float shifted = clamp(logits[base + i] - row_max, -60.0, 60.0);
        local_sum += exp(shifted);
    }
    float sg_sum = subgroupAdd(local_sum);
    if (subgroupElect()) lds_sum[sg_idx] = sg_sum;
    barrier();
    float sum_exp = 0.0;
    for (uint s = 0u; s < n_sgs; ++s) sum_exp += lds_sum[s];
    sum_exp = max(sum_exp, 1e-12);
    float inv_sum = 1.0 / sum_exp;

    // Pass 3: grad = coef * (softmax - one_hot).
    for (uint i = tid; i < num_classes; i += 256u) {
        float shifted = clamp(logits[base + i] - row_max, -60.0, 60.0);
        float sm = exp(shifted) * inv_sum;
        float g = sm;
        if (i == target) g -= 1.0;
        grad_logits[base + i] = c * g;
    }
}
