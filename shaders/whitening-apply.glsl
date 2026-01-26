#version 450

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Input data (batch, dim)
layout(set = 0, binding = 0) readonly buffer Input {
    float input_data[];
};

// Running mean (dim)
layout(set = 0, binding = 1) readonly buffer RunningMean {
    float mu[];
};

// Running variance (dim)
layout(set = 0, binding = 2) readonly buffer RunningVariance {
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
    float eps;           // Numerical stability constant
};

void main() {
    uint gID = gl_GlobalInvocationID.x;
    uint total_elements = batch_size * dim;
    
    if (gID >= total_elements) {
        return;
    }
    
    uint dim_idx = gID % dim;
    
    float x = input_data[gID];
    float mean_val = mu[dim_idx];
    float variance = var[dim_idx];
    
    // Whitening: (x - mu) / sqrt(var + eps)
    float centered = x - mean_val;
    float std_dev = sqrt(variance + eps);
    float whitened = centered / std_dev;
    
    output_data[gID] = whitened;
}
