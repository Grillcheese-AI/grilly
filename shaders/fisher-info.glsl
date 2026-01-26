#version 450

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Gradients from current batch (num_params)
layout(set = 0, binding = 0) readonly buffer Gradients {
    float grads[];
};

// Accumulated Fisher information (num_params)
layout(set = 0, binding = 1) buffer FisherInfo {
    float fisher[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint num_params;
    float momentum;      // EMA momentum (e.g., 0.9 for running estimate)
    uint use_ema;        // 1 = use EMA, 0 = accumulate
    uint reset_fisher;   // 1 = reset before accumulation, 0 = continue
};

void main() {
    uint gID = gl_GlobalInvocationID.x;
    
    if (gID >= num_params) {
        return;
    }
    
    float grad = grads[gID];
    float current_fisher = fisher[gID];
    
    // Reset if requested
    if (reset_fisher == 1) {
        current_fisher = 0.0;
    }
    
    // Fisher information: E[∇log p(θ)²] ≈ mean(gradient²)
    float grad_squared = grad * grad;
    
    if (use_ema == 1) {
        // Exponential moving average
        fisher[gID] = momentum * current_fisher + (1.0 - momentum) * grad_squared;
    } else {
        // Simple accumulation (divide by num_samples externally)
        fisher[gID] = current_fisher + grad_squared;
    }
}
