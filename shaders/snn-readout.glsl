#version 450

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Input spikes/activations (batch * seq, timesteps, features)
layout(set = 0, binding = 0) readonly buffer InputActivations {
    float activations[];
};

// Output (batch_size, seq_len/k, dim) or (batch_size, dim)
layout(set = 0, binding = 2) buffer Output {
    float output_data[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint batch_seq;       // batch_size * seq_len
    uint num_timesteps;   // Number of timesteps to average over
    uint features;        // Feature dimension
    uint reduction_type;  // 0 = mean, 1 = max, 2 = sum
};

void main() {
    uint gID = gl_GlobalInvocationID.x;
    uint total_samples = batch_seq * features;
    
    if (gID >= total_samples) {
        return;
    }
    
    // Decode indices
    uint sample_idx = gID / features;
    uint feat_idx = gID % features;
    
    // Base offset for this (sample, feature) across all timesteps
    uint base_offset = sample_idx * num_timesteps * features + feat_idx;
    
    float result = 0.0;
    
    if (reduction_type == 0) {
        // Mean reduction
        for (uint t = 0; t < num_timesteps; t++) {
            uint idx = base_offset + t * features;
            result += activations[idx];
        }
        result /= float(num_timesteps);
        
    } else if (reduction_type == 1) {
        // Max reduction
        result = -1e10; // Start with very negative number
        for (uint t = 0; t < num_timesteps; t++) {
            uint idx = base_offset + t * features;
            result = max(result, activations[idx]);
        }
        
    } else if (reduction_type == 2) {
        // Sum reduction
        for (uint t = 0; t < num_timesteps; t++) {
            uint idx = base_offset + t * features;
            result += activations[idx];
        }
    }
    
    // Write output
    uint out_idx = sample_idx * features + feat_idx;
    output_data[out_idx] = result;
}
