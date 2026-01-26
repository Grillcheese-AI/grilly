#version 450

layout (local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

// Input spikes (batch_size * seq_len, in_features)
layout(set = 0, binding = 0) readonly buffer InputSpikes {
    float spikes[];
};

// Synaptic weight matrix (out_features, in_features)
layout(set = 0, binding = 1) readonly buffer Weights {
    float W[];
};

// Bias vector (out_features)
layout(set = 0, binding = 2) readonly buffer Bias {
    float b[];
};

// Output currents (batch_size * seq_len, out_features)
layout(set = 0, binding = 3) buffer OutputCurrents {
    float currents[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint batch_time;      // batch_size * seq_len
    uint in_features;     // Input dimension
    uint out_features;    // Output dimension
    uint has_bias;        // 1 if bias exists, 0 otherwise
};

void main() {
    // Each thread computes one output element: currents[row][col]
    uint row = gl_GlobalInvocationID.y;  // Output sample index (0 to batch_time-1)
    uint col = gl_GlobalInvocationID.x;  // Output feature index (0 to out_features-1)
    
    if (row >= batch_time || col >= out_features) {
        return;
    }
    
    // Compute dot product: currents[row][col] = sum(spikes[row][k] * W[col][k])
    float sum = 0.0;
    
    for (uint k = 0; k < in_features; k++) {
        uint spike_idx = row * in_features + k;
        uint weight_idx = col * in_features + k;  // Row-major: W[col][k]
        
        sum += spikes[spike_idx] * W[weight_idx];
    }
    
    // Add bias if enabled
    if (has_bias == 1) {
        sum += b[col];
    }
    
    // Write output
    uint out_idx = row * out_features + col;
    currents[out_idx] = sum;
}
