#version 450

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Gradients w.r.t. parameters (num_params)
layout(set = 0, binding = 0) readonly buffer Gradients {
    float grads[];
};

// Fisher information (num_params)
layout(set = 0, binding = 1) readonly buffer FisherInfo {
    float fisher[];
};

// Updated gradients with Fisher-scaled learning (num_params)
layout(set = 0, binding = 2) buffer ScaledGradients {
    float scaled_grads[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint num_params;
    float base_lr;       // Base learning rate
    float epsilon;       // Small constant for stability
};

void main() {
    uint gID = gl_GlobalInvocationID.x;
    
    if (gID >= num_params) {
        return;
    }
    
    float grad = grads[gID];
    float F = fisher[gID];
    
    // Natural gradient: ∇_natural = F^(-1) * ∇
    // For diagonal approximation: scale by 1/(F + ε)
    float natural_grad = grad / (F + epsilon);
    
    // Apply learning rate
    scaled_grads[gID] = base_lr * natural_grad;
}
