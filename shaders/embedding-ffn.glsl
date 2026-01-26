#version 450

layout (local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

// Input (batch * seq, hidden_dim)
layout(set = 0, binding = 0) readonly buffer Input {
    float input_data[];
};

// FFN weights: W1 (intermediate_dim, hidden_dim)
layout(set = 0, binding = 1) readonly buffer W1 {
    float w1[];
};

// FFN bias: b1 (intermediate_dim)
layout(set = 0, binding = 2) readonly buffer B1 {
    float b1[];
};

// FFN weights: W2 (hidden_dim, intermediate_dim)
layout(set = 0, binding = 3) readonly buffer W2 {
    float w2[];
};

// FFN bias: b2 (hidden_dim)
layout(set = 0, binding = 4) readonly buffer B2 {
    float b2[];
};

// Intermediate buffer (batch * seq, intermediate_dim)
layout(set = 0, binding = 5) buffer Intermediate {
    float intermediate[];
};

// Output (batch * seq, hidden_dim)
layout(set = 0, binding = 6) buffer Output {
    float output_data[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint batch_seq;
    uint hidden_dim;
    uint intermediate_dim;
    uint activation_type;  // 0 = GELU, 1 = ReLU, 2 = SiLU
    uint pass_type;        // 0 = first linear + activation, 1 = second linear + residual
};

// GELU approximation (tanh version, matches PyTorch)
float gelu(float x) {
    const float sqrt_2_pi = 0.7978845608;  // sqrt(2/pi)
    const float coef = 0.044715;
    float inner = sqrt_2_pi * (x + coef * x * x * x);
    return 0.5 * x * (1.0 + tanh(inner));
}

// SiLU (Swish)
float silu(float x) {
    return x / (1.0 + exp(-x));
}

void main() {
    uint row = gl_GlobalInvocationID.y;
    uint col = gl_GlobalInvocationID.x;
    
    if (pass_type == 0) {
        // First linear: hidden_dim -> intermediate_dim + activation
        if (row >= batch_seq || col >= intermediate_dim) return;
        
        float sum = b1[col];
        for (uint k = 0; k < hidden_dim; k++) {
            uint in_idx = row * hidden_dim + k;
            uint w_idx = col * hidden_dim + k;
            sum += input_data[in_idx] * w1[w_idx];
        }
        
        // Apply activation
        float activated;
        if (activation_type == 0) {
            activated = gelu(sum);
        } else if (activation_type == 1) {
            activated = max(0.0, sum);  // ReLU
        } else {
            activated = silu(sum);
        }
        
        uint out_idx = row * intermediate_dim + col;
        intermediate[out_idx] = activated;
        
    } else {
        // Second linear: intermediate_dim -> hidden_dim + residual
        if (row >= batch_seq || col >= hidden_dim) return;
        
        float sum = b2[col];
        for (uint k = 0; k < intermediate_dim; k++) {
            uint in_idx = row * intermediate_dim + k;
            uint w_idx = col * intermediate_dim + k;
            sum += intermediate[in_idx] * w2[w_idx];
        }
        
        // Add residual connection
        uint out_idx = row * hidden_dim + col;
        output_data[out_idx] = sum + input_data[out_idx];
    }
}
