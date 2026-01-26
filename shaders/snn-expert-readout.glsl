#version 450

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Input activations from last layer (batch, time, hidden_dim)
layout(set = 0, binding = 0) readonly buffer InputActivations {
    float activations[];
};

// Readout weight matrix (output_dim, hidden_dim)
layout(set = 0, binding = 1) readonly buffer ReadoutWeights {
    float W[];
};

// Readout bias (output_dim)
layout(set = 0, binding = 2) readonly buffer ReadoutBias {
    float b[];
};

// Output predictions (batch, output_dim)
layout(set = 0, binding = 3) buffer Output {
    float output_data[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint batch_size;
    uint seq_len;
    uint hidden_dim;
    uint output_dim;
};

void main() {
    uint gID = gl_GlobalInvocationID.x;
    uint total_outputs = batch_size * output_dim;
    
    if (gID >= total_outputs) {
        return;
    }
    
    uint batch_idx = gID / output_dim;
    uint out_idx = gID % output_dim;
    
    // Step 1: Mean pooling over time for this (batch, hidden_feature)
    // Step 2: Linear projection to output
    
    float result = 0.0;
    
    for (uint h = 0; h < hidden_dim; h++) {
        // Compute mean over time for this hidden dimension
        float time_avg = 0.0;
        for (uint t = 0; t < seq_len; t++) {
            uint act_idx = batch_idx * seq_len * hidden_dim + t * hidden_dim + h;
            time_avg += activations[act_idx];
        }
        time_avg /= float(seq_len);
        
        // Apply weight
        uint w_idx = out_idx * hidden_dim + h;
        result += time_avg * W[w_idx];
    }
    
    // Add bias
    result += b[out_idx];
    
    output_data[gID] = result;
}
