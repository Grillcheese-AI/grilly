#version 450

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Input tensor
layout(set = 0, binding = 0) readonly buffer Input {
    float input_data[];
};

// Output normalized
layout(set = 0, binding = 1) buffer Output {
    float output_data[];
};

// Gamma (scale) parameters (features)
layout(set = 0, binding = 2) readonly buffer Gamma {
    float gamma[];
};

// Beta (shift) parameters (features)
layout(set = 0, binding = 3) readonly buffer Beta {
    float beta[];
};

// Mean buffer (batch * seq)
layout(set = 0, binding = 4) buffer MeanBuffer {
    float mean_vals[];
};

// Variance buffer (batch * seq)
layout(set = 0, binding = 5) buffer VarianceBuffer {
    float var_vals[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint batch_size;
    uint seq_len;
    uint features;
    float eps;            // Small constant for numerical stability (e.g., 1e-5)
    uint pass_type;       // 0 = compute mean, 1 = compute variance, 2 = normalize
};

void main() {
    uint gID = gl_GlobalInvocationID.x;
    
    if (pass_type == 0) {
        // Pass 1: Compute mean along feature dimension
        uint total_positions = batch_size * seq_len;
        if (gID >= total_positions) return;
        
        uint batch_idx = gID / seq_len;
        uint seq_idx = gID % seq_len;
        
        float sum = 0.0;
        for (uint f = 0; f < features; f++) {
            uint idx = batch_idx * seq_len * features + seq_idx * features + f;
            sum += input_data[idx];
        }
        mean_vals[gID] = sum / float(features);
        
    } else if (pass_type == 1) {
        // Pass 2: Compute variance
        uint total_positions = batch_size * seq_len;
        if (gID >= total_positions) return;
        
        uint batch_idx = gID / seq_len;
        uint seq_idx = gID % seq_len;
        float mean = mean_vals[gID];
        
        float sum_sq = 0.0;
        for (uint f = 0; f < features; f++) {
            uint idx = batch_idx * seq_len * features + seq_idx * features + f;
            float diff = input_data[idx] - mean;
            sum_sq += diff * diff;
        }
        var_vals[gID] = sum_sq / float(features);
        
    } else if (pass_type == 2) {
        // Pass 3: Normalize
        uint total_elements = batch_size * seq_len * features;
        if (gID >= total_elements) return;
        
        uint batch_idx = gID / (seq_len * features);
        uint remainder = gID % (seq_len * features);
        uint seq_idx = remainder / features;
        uint feat_idx = remainder % features;
        
        uint pos_idx = batch_idx * seq_len + seq_idx;
        float mean = mean_vals[pos_idx];
        float variance = var_vals[pos_idx];
        float std = sqrt(variance + eps);
        
        float x = input_data[gID];
        float normalized = (x - mean) / std;
        
        // Apply affine transformation
        float g = gamma[feat_idx];
        float b = beta[feat_idx];
        
        output_data[gID] = g * normalized + b;
    }
}
