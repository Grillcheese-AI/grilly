#version 450

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Input hidden states (batch, seq_len, hidden_dim)
layout(set = 0, binding = 0) readonly buffer Input {
    float input_data[];
};

// Attention mask (batch, seq_len) - 1.0 for valid tokens, 0.0 for padding
layout(set = 0, binding = 1) readonly buffer Mask {
    float mask[];
};

// Output pooled embedding (batch, hidden_dim)
layout(set = 0, binding = 2) buffer Output {
    float output_data[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint batch_size;
    uint seq_len;
    uint hidden_dim;
    uint pool_type;  // 0 = mean, 1 = cls (first token), 2 = max
};

void main() {
    uint gID = gl_GlobalInvocationID.x;
    
    // Each thread handles one (batch, dim) pair
    uint total_elements = batch_size * hidden_dim;
    if (gID >= total_elements) {
        return;
    }
    
    uint batch_idx = gID / hidden_dim;
    uint dim_idx = gID % hidden_dim;
    
    if (pool_type == 1) {
        // CLS pooling: just take first token
        uint cls_idx = batch_idx * seq_len * hidden_dim + dim_idx;
        output_data[gID] = input_data[cls_idx];
        return;
    }
    
    if (pool_type == 2) {
        // Max pooling
        float max_val = -1e30;
        for (uint s = 0; s < seq_len; s++) {
            uint mask_idx = batch_idx * seq_len + s;
            if (mask[mask_idx] > 0.5) {
                uint input_idx = batch_idx * seq_len * hidden_dim + s * hidden_dim + dim_idx;
                max_val = max(max_val, input_data[input_idx]);
            }
        }
        output_data[gID] = max_val;
        return;
    }
    
    // Mean pooling (default)
    float sum = 0.0;
    float count = 0.0;
    
    for (uint s = 0; s < seq_len; s++) {
        uint mask_idx = batch_idx * seq_len + s;
        float m = mask[mask_idx];
        
        if (m > 0.5) {
            uint input_idx = batch_idx * seq_len * hidden_dim + s * hidden_dim + dim_idx;
            sum += input_data[input_idx];
            count += 1.0;
        }
    }
    
    // Avoid division by zero
    if (count > 0.0) {
        output_data[gID] = sum / count;
    } else {
        output_data[gID] = 0.0;
    }
}
