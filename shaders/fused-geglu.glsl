#version 450

layout (local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

// Input (batch_seq, input_dim)
layout(set = 0, binding = 0) readonly buffer Input {
    float input_data[];
};

// Gate weights (hidden_dim, input_dim)
layout(set = 0, binding = 1) readonly buffer WGate {
    float W_gate[];
};

// Up weights (hidden_dim, input_dim)
layout(set = 0, binding = 2) readonly buffer WUp {
    float W_up[];
};

// Gate bias (hidden_dim)
layout(set = 0, binding = 3) readonly buffer BGate {
    float b_gate[];
};

// Up bias (hidden_dim)
layout(set = 0, binding = 4) readonly buffer BUp {
    float b_up[];
};

// Output (batch_seq, hidden_dim)
layout(set = 0, binding = 5) buffer Output {
    float output_data[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint batch_seq;
    uint input_dim;
    uint hidden_dim;
    uint has_bias;
};

// GELU activation (approximate)
float gelu(float x) {
    // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    float sqrt_2_over_pi = 0.7978845608028654;
    float coeff = 0.044715;
    return 0.5 * x * (1.0 + tanh(sqrt_2_over_pi * (x + coeff * x * x * x)));
}

void main() {
    uint row = gl_GlobalInvocationID.y;  // batch index
    uint col = gl_GlobalInvocationID.x;  // hidden dim index

    if (row >= batch_seq || col >= hidden_dim) return;

    // Compute gate = x @ W_gate^T + b_gate
    float gate = 0.0;
    for (uint k = 0; k < input_dim; k++) {
        uint x_idx = row * input_dim + k;
        uint w_idx = col * input_dim + k;
        gate += input_data[x_idx] * W_gate[w_idx];
    }
    if (has_bias != 0) {
        gate += b_gate[col];
    }

    // Compute up = x @ W_up^T + b_up
    float up = 0.0;
    for (uint k = 0; k < input_dim; k++) {
        uint x_idx = row * input_dim + k;
        uint w_idx = col * input_dim + k;
        up += input_data[x_idx] * W_up[w_idx];
    }
    if (has_bias != 0) {
        up += b_up[col];
    }

    // GeGLU: gate * GELU(up)
    uint out_idx = row * hidden_dim + col;
    output_data[out_idx] = gate * gelu(up);
}
