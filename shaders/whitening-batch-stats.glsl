#version 450
#extension GL_EXT_shader_atomic_float : require

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Input data batch (batch_size, dim)
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

// Batch statistics buffers (dim) - for atomic accumulation
layout(set = 0, binding = 3) buffer BatchMean {
    float batch_mu[];
};

layout(set = 0, binding = 4) buffer BatchVar {
    float batch_var[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint batch_size;
    uint dim;
    float momentum;
    uint pass_type;      // 0 = accumulate batch stats, 1 = update running stats
};

void main() {
    uint gID = gl_GlobalInvocationID.x;
    
    if (pass_type == 0) {
        // Pass 1: Accumulate batch statistics
        uint total_elements = batch_size * dim;
        if (gID >= total_elements) return;
        
        uint batch_idx = gID / dim;
        uint dim_idx = gID % dim;
        
        float x = input_data[gID];
        
        // Atomic add for batch mean
        atomicAdd(batch_mu[dim_idx], x);
        
    } else if (pass_type == 1) {
        // Pass 2: Update running statistics
        if (gID >= dim) return;
        
        // Compute batch mean
        float mean_batch = batch_mu[gID] / float(batch_size);
        
        // Compute batch variance
        float var_batch = 0.0;
        for (uint b = 0; b < batch_size; b++) {
            uint idx = b * dim + gID;
            float x = input_data[idx];
            float diff = x - mean_batch;
            var_batch += diff * diff;
        }
        var_batch /= float(batch_size);
        
        // Update running statistics
        mu[gID] = mu[gID] * (1.0 - momentum) + mean_batch * momentum;
        var[gID] = var[gID] * (1.0 - momentum) + var_batch * momentum;
        
        // Clear batch buffers for next iteration
        batch_mu[gID] = 0.0;
        batch_var[gID] = 0.0;
    }
}
