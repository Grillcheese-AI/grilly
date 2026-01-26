#version 450

layout (local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

// Input spikes (batch * time, in_features)
layout(set = 0, binding = 0) readonly buffer InputSpikes {
    float spikes[];
};

// Weight matrix (out_features, in_features)
layout(set = 0, binding = 1) readonly buffer Weights {
    float W[];
};

// Output (batch * time, out_features)
layout(set = 0, binding = 2) buffer Output {
    float output_data[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint batch_time;      // batch_size * seq_len
    uint in_features;
    uint out_features;
    float scale_factor;   // sqrt(in_features) if scaling enabled, else 1.0
    uint use_scaling;     // 1 = apply scaling, 0 = no scaling
};

void main() {
    uint row = gl_GlobalInvocationID.y;  // Output sample
    uint col = gl_GlobalInvocationID.x;  // Output feature
    
    if (row >= batch_time || col >= out_features) {
        return;
    }
    
    // Matrix multiplication: output[row][col] = sum(spikes[row][k] * W[col][k])
    float sum = 0.0;
    
    for (uint k = 0; k < in_features; k++) {
        uint spike_idx = row * in_features + k;
        uint weight_idx = col * in_features + k;
        sum += spikes[spike_idx] * W[weight_idx];
    }
    
    // Apply scaling for numerical stability
    if (use_scaling == 1) {
        sum /= scale_factor;
    }
    
    uint out_idx = row * out_features + col;
    output_data[out_idx] = sum;
}
