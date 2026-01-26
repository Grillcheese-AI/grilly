#version 450

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Input spikes
layout(set = 0, binding = 0) readonly buffer InputSpikes {
    float spikes[];
};

// Output normalized
layout(set = 0, binding = 1) buffer Output {
    float output_data[];
};

// Max values buffer
layout(set = 0, binding = 2) buffer MaxValues {
    float max_vals[];
};

// Sum of exponentials buffer
layout(set = 0, binding = 3) buffer SumExp {
    float sum_exp[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint batch_size;
    uint seq_len;
    uint features;
    float temperature;    // Temperature scaling (default 1.0)
    uint pass_type;       // 0 = max, 1 = sum, 2 = normalize
};

void main() {
    uint gID = gl_GlobalInvocationID.x;
    
    if (pass_type == 0) {
        // Pass 1: Compute max with temperature scaling
        uint total_positions = batch_size * seq_len;
        if (gID >= total_positions) return;
        
        uint batch_idx = gID / seq_len;
        uint seq_idx = gID % seq_len;
        
        float max_val = -1e10;
        for (uint f = 0; f < features; f++) {
            uint idx = batch_idx * seq_len * features + seq_idx * features + f;
            float scaled = spikes[idx] / temperature;
            max_val = max(max_val, scaled);
        }
        max_vals[gID] = max_val;
        
    } else if (pass_type == 1) {
        // Pass 2: Compute sum of exp
        uint total_positions = batch_size * seq_len;
        if (gID >= total_positions) return;
        
        uint batch_idx = gID / seq_len;
        uint seq_idx = gID % seq_len;
        float max_val = max_vals[gID];
        
        float sum = 0.0;
        for (uint f = 0; f < features; f++) {
            uint idx = batch_idx * seq_len * features + seq_idx * features + f;
            float scaled = spikes[idx] / temperature;
            sum += exp(scaled - max_val);
        }
        sum_exp[gID] = sum;
        
    } else if (pass_type == 2) {
        // Pass 3: Normalize (spike-based softmax)
        uint total_elements = batch_size * seq_len * features;
        if (gID >= total_elements) return;
        
        uint batch_idx = gID / (seq_len * features);
        uint remainder = gID % (seq_len * features);
        uint seq_idx = remainder / features;
        
        uint pos_idx = batch_idx * seq_len + seq_idx;
        float max_val = max_vals[pos_idx];
        float sum = sum_exp[pos_idx];
        
        float scaled = spikes[gID] / temperature;
        output_data[gID] = exp(scaled - max_val) / sum;
    }
}
