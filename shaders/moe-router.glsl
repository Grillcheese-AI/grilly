#version 450

// MoE Router: mean(x) → logits → softmax → router weights
//
// Three passes (controlled by push constant `pass`):
//   Pass 0: Compute mean(x) across seq_len → x_mean (d_model,)
//   Pass 1: Compute logits = router_W @ x_mean + router_b → (n_experts,)
//   Pass 2: Stable softmax over logits → router_weights (n_experts,)
//
// Scratch layout: [x_mean (d) | logits (E) | weights (E)]

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer Input {
    float x[];
};

layout(set = 0, binding = 1) readonly buffer RouterW {
    float rw[];
};

layout(set = 0, binding = 2) readonly buffer RouterB {
    float rb[];
};

layout(set = 0, binding = 3) buffer Scratch {
    float scratch[];
};

layout(push_constant) uniform PushConsts {
    uint seq_len;
    uint d_model;
    uint n_experts;
    uint pass;
};

void main() {
    uint gid = gl_GlobalInvocationID.x;

    if (pass == 0u) {
        // Pass 0: mean reduction
        if (gid >= d_model) return;
        float sum = 0.0;
        for (uint s = 0; s < seq_len; s++) {
            sum += x[s * d_model + gid];
        }
        scratch[gid] = sum / float(seq_len);
    }
    else if (pass == 1u) {
        // Pass 1: logits = rw @ x_mean + rb
        if (gid >= n_experts) return;
        float dot = 0.0;
        for (uint k = 0; k < d_model; k++) {
            dot += rw[gid * d_model + k] * scratch[k];
        }
        scratch[d_model + gid] = dot + rb[gid];
    }
    else if (pass == 2u) {
        // Pass 2: numerically stable softmax (single thread, n_experts is small)
        if (gid != 0u) return;
        uint logits_off = d_model;
        uint weights_off = d_model + n_experts;

        // Find max for numerical stability
        float max_val = scratch[logits_off];
        for (uint e = 1; e < n_experts; e++) {
            max_val = max(max_val, scratch[logits_off + e]);
        }

        // Exp with max subtraction + sum
        float sum_exp = 0.0;
        for (uint e = 0; e < n_experts; e++) {
            float ev = exp(scratch[logits_off + e] - max_val);
            scratch[weights_off + e] = ev;
            sum_exp += ev;
        }

        // Normalize
        float inv_sum = 1.0 / (sum_exp + 1e-8);
        for (uint e = 0; e < n_experts; e++) {
            scratch[weights_off + e] *= inv_sum;
        }
    }
}
