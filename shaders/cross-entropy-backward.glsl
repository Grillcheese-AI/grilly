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

// Shared memory for parallel reduction
shared float s_data[256];

void main() {
    uint batch_idx = gl_WorkGroupID.x;
    uint tid = gl_LocalInvocationID.x;

    if (batch_idx >= batch_size) return;

    uint base_idx = batch_idx * num_classes;

    // Step 1: Parallel max reduction — all 256 threads participate
    float local_max = -1e30;
    for (uint i = tid; i < num_classes; i += 256) {
        local_max = max(local_max, logits[base_idx + i]);
    }
    s_data[tid] = local_max;
    barrier();

    // Tree reduction for max (log2(256) = 8 steps)
    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] = max(s_data[tid], s_data[tid + s]);
        }
        barrier();
    }
    float max_val = s_data[0];
    barrier();

    // Step 2: Parallel sum of exp(x - max)
    float local_sum = 0.0;
    for (uint i = tid; i < num_classes; i += 256) {
        local_sum += exp(logits[base_idx + i] - max_val);
    }
    s_data[tid] = local_sum;
    barrier();

    // Tree reduction for sum
    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
        }
        barrier();
    }
    float sum_exp = s_data[0];
    barrier();

    // Step 3: Compute gradient = softmax - one_hot (all threads)
    uint target_class = uint(targets[batch_idx]);
    float inv_sum = 1.0 / max(sum_exp, 1e-12);

    for (uint i = tid; i < num_classes; i += 256) {
        float softmax_val = exp(logits[base_idx + i] - max_val) * inv_sum;
        float grad = softmax_val;
        if (i == target_class) {
            grad -= 1.0;
        }
        grad_logits[base_idx + i] = grad;
    }
}
