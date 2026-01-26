#version 450

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Input spikes
layout(set = 0, binding = 0) readonly buffer Input {
    float input_data[];
};

// Output normalized
layout(set = 0, binding = 1) buffer Output {
    float output_data[];
};

// Gamma scaling parameters (per feature)
layout(set = 0, binding = 2) readonly buffer Gamma {
    float gamma[];
};

// RMS values buffer (batch * seq)
layout(set = 0, binding = 3) buffer RMSValues {
    float rms_vals[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint batch_size;
    uint seq_len;
    uint features;
    float eps;            // Small constant for numerical stability (e.g., 1e-6)
    uint pass_type;       // 0 = compute RMS, 1 = normalize
};

void main() {
    uint gID = gl_GlobalInvocationID.x;
    
    if (pass_type == 0) {
        // Pass 1: Compute RMS along feature dimension
        uint total_positions = batch_size * seq_len;
        if (gID >= total_positions) return;
        
        uint batch_idx = gID / seq_len;
        uint seq_idx = gID % seq_len;
        
        // Compute mean of squares
        float mean_sq = 0.0;
        for (uint f = 0; f < features; f++) {
            uint idx = batch_idx * seq_len * features + seq_idx * features + f;
            float val = input_data[idx];
            mean_sq += val * val;
        }
        mean_sq /= float(features);
        
        // RMS = sqrt(mean(x^2) + eps)
        float rms = sqrt(mean_sq + eps);
        rms_vals[gID] = rms;
        
    } else if (pass_type == 1) {
        // Pass 2: Normalize and scale with gamma
        uint total_elements = batch_size * seq_len * features;
        if (gID >= total_elements) return;
        
        uint batch_idx = gID / (seq_len * features);
        uint remainder = gID % (seq_len * features);
        uint seq_idx = remainder / features;
        uint feat_idx = remainder % features;
        
        uint pos_idx = batch_idx * seq_len + seq_idx;
        float rms = rms_vals[pos_idx];
        float g = gamma[feat_idx];
        
        float val = input_data[gID];
        output_data[gID] = (val / rms) * g;
    }
}
