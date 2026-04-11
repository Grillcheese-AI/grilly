#version 450
#extension GL_KHR_shader_subgroup_arithmetic : require

// Fused Knowledge Distillation + Cross-Entropy Loss + Gradient Shader
//
// Computes in ONE dispatch per token:
//   1. Softmax(student_logits/T) and Softmax(teacher_logits/T)
//   2. KL divergence loss: T^2 * sum(t * log(t/s))
//   3. Cross-entropy loss: -log(s[label])
//   4. Combined gradient: (1-alpha)*grad_CE + alpha*grad_KD
//
// 1 Workgroup (1024 threads) = 1 Token
// gl_WorkGroupID.x = Token Index
//
// Uses subgroup arithmetic for fast reductions (no shared memory barriers
// for intra-subgroup ops).

layout(local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;

layout(std430, set = 0, binding = 0) readonly buffer StudentBuf { float S_Logits[]; };
layout(std430, set = 0, binding = 1) readonly buffer TeacherBuf { float T_Logits[]; };
layout(std430, set = 0, binding = 2) readonly buffer LabelBuf { int Labels[]; };
layout(std430, set = 0, binding = 3) writeonly buffer GradBuf { float GradOut[]; };
layout(std430, set = 0, binding = 4) writeonly buffer LossBuf { float LossOut[]; };

layout(push_constant) uniform Params {
    uint vocab_size;
    float temperature;
    float alpha;        // KD weight: loss = (1-alpha)*CE + alpha*KD
} p;

shared float shared_s_max[32];
shared float shared_t_max[32];
shared float shared_s_sum[32];
shared float shared_t_sum[32];
shared float shared_ce_sum[32];
shared float shared_ce_max[32];

void main() {
    uint token_idx = gl_WorkGroupID.x;
    uint local_id = gl_LocalInvocationID.x;
    uint subgroup_id = gl_SubgroupID;
    uint lane_id = gl_SubgroupInvocationID;

    // ── PASS 1: Find Max Logit (numerical stability) ────────────────────
    float my_s_max = -1e30;
    float my_t_max = -1e30;
    float my_ce_max = -1e30;

    for (uint v = local_id; v < p.vocab_size; v += gl_WorkGroupSize.x) {
        uint flat_idx = token_idx * p.vocab_size + v;
        float s_val = S_Logits[flat_idx];
        float t_val = T_Logits[flat_idx];

        my_s_max = max(my_s_max, s_val / p.temperature);
        my_t_max = max(my_t_max, t_val / p.temperature);
        my_ce_max = max(my_ce_max, s_val);
    }

    my_s_max = subgroupMax(my_s_max);
    my_t_max = subgroupMax(my_t_max);
    my_ce_max = subgroupMax(my_ce_max);

    if (lane_id == 0) {
        shared_s_max[subgroup_id] = my_s_max;
        shared_t_max[subgroup_id] = my_t_max;
        shared_ce_max[subgroup_id] = my_ce_max;
    }
    barrier();

    if (local_id == 0) {
        float gs = -1e30, gt = -1e30, gc = -1e30;
        for (int i = 0; i < gl_NumSubgroups; i++) {
            gs = max(gs, shared_s_max[i]);
            gt = max(gt, shared_t_max[i]);
            gc = max(gc, shared_ce_max[i]);
        }
        shared_s_max[0] = gs;
        shared_t_max[0] = gt;
        shared_ce_max[0] = gc;
    }
    barrier();

    float row_s_max = shared_s_max[0];
    float row_t_max = shared_t_max[0];
    float row_ce_max = shared_ce_max[0];

    // ── PASS 2: Compute Softmax Denominators ────────────────────────────
    float my_s_sum = 0.0;
    float my_t_sum = 0.0;
    float my_ce_sum = 0.0;
    float my_kl_loss = 0.0;

    for (uint v = local_id; v < p.vocab_size; v += gl_WorkGroupSize.x) {
        uint flat_idx = token_idx * p.vocab_size + v;
        float s_val = S_Logits[flat_idx];
        float t_val = T_Logits[flat_idx];

        my_s_sum += exp((s_val / p.temperature) - row_s_max);
        my_t_sum += exp((t_val / p.temperature) - row_t_max);
        my_ce_sum += exp(s_val - row_ce_max);
    }

    my_s_sum = subgroupAdd(my_s_sum);
    my_t_sum = subgroupAdd(my_t_sum);
    my_ce_sum = subgroupAdd(my_ce_sum);

    if (lane_id == 0) {
        shared_s_sum[subgroup_id] = my_s_sum;
        shared_t_sum[subgroup_id] = my_t_sum;
        shared_ce_sum[subgroup_id] = my_ce_sum;
    }
    barrier();

    if (local_id == 0) {
        float gs = 0.0, gt = 0.0, gc = 0.0;
        for (int i = 0; i < gl_NumSubgroups; i++) {
            gs += shared_s_sum[i];
            gt += shared_t_sum[i];
            gc += shared_ce_sum[i];
        }
        shared_s_sum[0] = gs;
        shared_t_sum[0] = gt;
        shared_ce_sum[0] = gc;
    }
    barrier();

    float row_s_sum = shared_s_sum[0] + 1e-8;
    float row_t_sum = shared_t_sum[0] + 1e-8;
    float row_ce_sum = shared_ce_sum[0] + 1e-8;

    // ── PASS 3: Gradients + Loss ────────────────────────────────────────
    int label = Labels[token_idx];

    for (uint v = local_id; v < p.vocab_size; v += gl_WorkGroupSize.x) {
        uint flat_idx = token_idx * p.vocab_size + v;
        float s_val = S_Logits[flat_idx];
        float t_val = T_Logits[flat_idx];

        // Softmax probabilities
        float s_prob_kd = exp((s_val / p.temperature) - row_s_max) / row_s_sum;
        float t_prob_kd = exp((t_val / p.temperature) - row_t_max) / row_t_sum;
        float s_prob_ce = exp(s_val - row_ce_max) / row_ce_sum;

        // KL divergence accumulation
        float t_log = log(max(t_prob_kd, 1e-7));
        float s_log = log(max(s_prob_kd, 1e-7));
        my_kl_loss += t_prob_kd * (t_log - s_log);

        // Gradients
        float grad_kd = (s_prob_kd - t_prob_kd) * p.temperature;
        float grad_ce = s_prob_ce;
        if (v == label) {
            grad_ce -= 1.0;
        }

        GradOut[flat_idx] = (1.0 - p.alpha) * grad_ce + p.alpha * grad_kd;
    }

    // ── Reduce KL loss across threads ───────────────────────────────────
    my_kl_loss = subgroupAdd(my_kl_loss);
    if (lane_id == 0) shared_s_sum[subgroup_id] = my_kl_loss;
    barrier();

    if (local_id == 0) {
        float token_kl = 0.0;
        for (int i = 0; i < gl_NumSubgroups; i++) {
            token_kl += shared_s_sum[i];
        }
        token_kl *= (p.temperature * p.temperature);

        // CE loss for this token
        float token_ce = 0.0;
        if (label >= 0 && label < p.vocab_size) {
            float target_logit = S_Logits[token_idx * p.vocab_size + label];
            float target_prob = exp(target_logit - row_ce_max) / row_ce_sum;
            token_ce = -log(max(target_prob, 1e-8));
        }

        LossOut[token_idx] = (1.0 - p.alpha) * token_ce + p.alpha * token_kl;
    }
}
