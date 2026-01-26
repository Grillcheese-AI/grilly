#version 450

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Current parameters (num_params)
layout(set = 0, binding = 0) readonly buffer CurrentParams {
    float current_params[];
};

// Stored optimal parameters from previous task (num_params)
layout(set = 0, binding = 1) readonly buffer OptimalParams {
    float optimal_params[];
};

// Fisher information scores (num_params)
layout(set = 0, binding = 2) readonly buffer FisherInfo {
    float fisher[];
};

// Output EWC penalty (num_params) - for reduction later
layout(set = 0, binding = 3) buffer EWCPenalty {
    float penalty[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint num_params;
    float lambda;        // EWC regularization strength
};

void main() {
    uint gID = gl_GlobalInvocationID.x;
    
    if (gID >= num_params) {
        return;
    }
    
    float theta_current = current_params[gID];
    float theta_optimal = optimal_params[gID];
    float F = fisher[gID];
    
    // EWC penalty: (λ/2) * F * (θ - θ*)²
    float diff = theta_current - theta_optimal;
    float param_penalty = 0.5 * lambda * F * diff * diff;
    
    penalty[gID] = param_penalty;
}
