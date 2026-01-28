#version 450

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Logits (batch_size, num_classes)
layout(set = 0, binding = 0) readonly buffer Logits {
    float logits[];
};

// Target class indices (batch_size,) as floats
layout(set = 0, binding = 1) readonly buffer Targets {
    float targets[];
};

// Gradient w.r.t. logits (batch_size, num_classes)
layout(set = 0, binding = 2) buffer GradLogits {
    float grad_logits[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint batch_size;
    uint num_classes;
};

// Shared memory for softmax computation
shared float s_max;
shared float s_sum;

void main() {
    uint batch_idx = gl_WorkGroupID.x;
    uint class_idx = gl_LocalInvocationID.x;

    if (batch_idx >= batch_size) {
        return;
    }

    uint base_idx = batch_idx * num_classes;

    // Step 1: Find max for numerical stability (reduction)
    float local_max = -1e30;
    for (uint i = class_idx; i < num_classes; i += gl_WorkGroupSize.x) {
        float val = logits[base_idx + i];
        local_max = max(local_max, val);
    }

    // Reduce to find global max (simplified - assumes workgroup size >= num_classes)
    if (class_idx == 0) {
        float max_val = -1e30;
        for (uint i = 0; i < num_classes; i++) {
            max_val = max(max_val, logits[base_idx + i]);
        }
        s_max = max_val;
    }
    barrier();

    // Step 2: Compute exp(x - max) and sum
    if (class_idx == 0) {
        float sum_exp = 0.0;
        for (uint i = 0; i < num_classes; i++) {
            sum_exp += exp(logits[base_idx + i] - s_max);
        }
        s_sum = sum_exp;
    }
    barrier();

    // Step 3: Compute softmax and gradient
    // Gradient of cross-entropy with softmax: softmax - one_hot(target)
    uint target_class = uint(targets[batch_idx]);

    for (uint i = class_idx; i < num_classes; i += gl_WorkGroupSize.x) {
        float softmax_val = exp(logits[base_idx + i] - s_max) / s_sum;

        // Gradient: softmax - (1 if i == target else 0)
        float grad = softmax_val;
        if (i == target_class) {
            grad -= 1.0;
        }

        grad_logits[base_idx + i] = grad;
    }
}
