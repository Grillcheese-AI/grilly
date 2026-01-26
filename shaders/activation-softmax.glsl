#version 450

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Input tensor
layout(set = 0, binding = 0) readonly buffer Input {
    float input_data[];
};

// Output tensor (in-place if same buffer)
layout(set = 0, binding = 1) buffer Output {
    float output_data[];
};

// Temporary buffer for max values (one per sequence position)
layout(set = 0, binding = 2) buffer MaxValues {
    float max_vals[];
};

// Temporary buffer for sum of exponentials
layout(set = 0, binding = 3) buffer SumExp {
    float sum_exp[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint batch_size;
    uint seq_len;
    uint features;
    uint pass_type;  // 0 = compute max, 1 = compute sum, 2 = normalize
    uint dim;        // Dimension to normalize over (typically features)
};

void main() {
    uint gID = gl_GlobalInvocationID.x;
    
    if (pass_type == 0) {
        // Pass 1: Compute max along feature dimension for numerical stability
        uint total_positions = batch_size * seq_len;
        if (gID >= total_positions) return;
        
        uint batch_idx = gID / seq_len;
        uint seq_idx = gID % seq_len;
        
        float max_val = -1e10;
        for (uint f = 0; f < features; f++) {
            uint idx = batch_idx * seq_len * features + seq_idx * features + f;
            max_val = max(max_val, input_data[idx]);
        }
        max_vals[gID] = max_val;
        
    } else if (pass_type == 1) {
        // Pass 2: Compute sum of exp(x - max)
        uint total_positions = batch_size * seq_len;
        if (gID >= total_positions) return;
        
        uint batch_idx = gID / seq_len;
        uint seq_idx = gID % seq_len;
        float max_val = max_vals[gID];
        
        float sum = 0.0;
        for (uint f = 0; f < features; f++) {
            uint idx = batch_idx * seq_len * features + seq_idx * features + f;
            sum += exp(input_data[idx] - max_val);
        }
        sum_exp[gID] = sum;
        
    } else if (pass_type == 2) {
        // Pass 3: Normalize
        uint total_elements = batch_size * seq_len * features;
        if (gID >= total_elements) return;
        
        uint batch_idx = gID / (seq_len * features);
        uint remainder = gID % (seq_len * features);
        uint seq_idx = remainder / features;
        
        uint pos_idx = batch_idx * seq_len + seq_idx;
        float max_val = max_vals[pos_idx];
        float sum = sum_exp[pos_idx];
        
        float val = input_data[gID];
        output_data[gID] = exp(val - max_val) / sum;
    }
}
