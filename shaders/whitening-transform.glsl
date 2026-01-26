#version 450

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Input data (batch, dim)
layout(set = 0, binding = 0) readonly buffer Input {
    float input_data[];
};

// Running mean (dim) - read and write
layout(set = 0, binding = 1) buffer RunningMean {
    float mu[];
};

// Running variance (dim) - read and write
layout(set = 0, binding = 2) buffer RunningVariance {
    float var[];
};

// Whitened output (batch, dim)
layout(set = 0, binding = 3) buffer Output {
    float output_data[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint batch_size;
    uint dim;
    float momentum;      // Momentum for running stats (e.g., 0.01)
    float eps;           // Small constant for numerical stability (e.g., 1e-6)
};

void main() {
    uint gID = gl_GlobalInvocationID.x;
    uint total_elements = batch_size * dim;
    
    if (gID >= total_elements) {
        return;
    }
    
    uint batch_idx = gID / dim;
    uint dim_idx = gID % dim;
    
    float x = input_data[gID];
    
    // Update running mean (only first batch item updates stats)
    if (batch_idx == 0) {
        // mu = mu * (1 - momentum) + x * momentum
        float current_mu = mu[dim_idx];
        float new_mu = current_mu * (1.0 - momentum) + x * momentum;
        mu[dim_idx] = new_mu;
    }
    
    // Get current mean for whitening
    float mean_val = mu[dim_idx];
    
    // Calculate centered data
    float centered = x - mean_val;
    
    // Update running variance (only first batch item updates stats)
    if (batch_idx == 0) {
        // var = var * (1 - momentum) + centered^2 * momentum
        float current_var = var[dim_idx];
        float new_var = current_var * (1.0 - momentum) + centered * centered * momentum;
        var[dim_idx] = new_var;
    }
    
    // Get current variance for whitening
    float variance = var[dim_idx];
    
    // Apply whitening: (x - mu) / sqrt(var + eps)
    float std_dev = sqrt(variance + eps);
    float whitened = centered / std_dev;
    
    output_data[gID] = whitened;
}
