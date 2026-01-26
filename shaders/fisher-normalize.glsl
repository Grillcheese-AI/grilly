#version 450

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Accumulated Fisher values (num_params)
layout(set = 0, binding = 0) buffer FisherInfo {
    float fisher[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint num_params;
    uint num_samples;    // Number of samples accumulated
    float epsilon;       // Small constant for numerical stability (e.g., 1e-8)
};

void main() {
    uint gID = gl_GlobalInvocationID.x;
    
    if (gID >= num_params) {
        return;
    }
    
    // Normalize by number of samples
    float normalized = fisher[gID] / float(num_samples);
    
    // Add epsilon for numerical stability
    fisher[gID] = normalized + epsilon;
}
