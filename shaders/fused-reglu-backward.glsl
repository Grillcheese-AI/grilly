#version 450

layout (local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

// Gradient w.r.t. output (batch_seq, hidden_dim)
layout(set = 0, binding = 0) readonly buffer GradOutput {
    float grad_output[];
};

// Input (batch_seq, input_dim)
layout(set = 0, binding = 1) readonly buffer Input {
    float input_data[];
};

// Gate weights (hidden_dim, input_dim)
layout(set = 0, binding = 2) readonly buffer WGate {
    float W_gate[];
};

// Up weights (hidden_dim, input_dim)
layout(set = 0, binding = 3) readonly buffer WUp {
    float W_up[];
};

// Gate pre-activation (batch_seq, hidden_dim)
layout(set = 0, binding = 4) readonly buffer GateCache {
    float gate_cache[];
};

// Up pre-activation (batch_seq, hidden_dim)
layout(set = 0, binding = 5) readonly buffer UpCache {
    float up_cache[];
};

// Gradient w.r.t. input (batch_seq, input_dim)
layout(set = 0, binding = 6) buffer GradInput {
    float grad_input[];
};

// Gradient w.r.t. W_gate (hidden_dim, input_dim)
layout(set = 0, binding = 7) buffer GradWGate {
    float grad_W_gate[];
};

// Gradient w.r.t. W_up (hidden_dim, input_dim)
layout(set = 0, binding = 8) buffer GradWUp {
    float grad_W_up[];
};

// Gradient w.r.t. b_gate (hidden_dim)
layout(set = 0, binding = 9) buffer GradBGate {
    float grad_b_gate[];
};

// Gradient w.r.t. b_up (hidden_dim)
layout(set = 0, binding = 10) buffer GradBUp {
    float grad_b_up[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint batch_seq;
    uint input_dim;
    uint hidden_dim;
    uint pass_type;
};

void main() {
    uint row = gl_GlobalInvocationID.y;
    uint col = gl_GlobalInvocationID.x;

    if (pass_type == 0) {
        // Compute gradient w.r.t. input
        if (row >= batch_seq || col >= input_dim) return;

        float sum = 0.0;
        for (uint h = 0; h < hidden_dim; h++) {
            uint cache_idx = row * hidden_dim + h;
            float gate = gate_cache[cache_idx];
            float up = up_cache[cache_idx];

            float grad_out = grad_output[cache_idx];
            // ReGLU: output = gate * ReLU(up)
            float relu_up = max(0.0, up);
            float grad_gate = grad_out * relu_up;
            float grad_up = grad_out * gate * (up > 0.0 ? 1.0 : 0.0);

            uint w_idx = h * input_dim + col;
            sum += grad_gate * W_gate[w_idx] + grad_up * W_up[w_idx];
        }

        uint out_idx = row * input_dim + col;
        grad_input[out_idx] = sum;

    } else if (pass_type == 1) {
        // Compute gradient w.r.t. weights
        if (row >= hidden_dim || col >= input_dim) return;

        float sum_gate = 0.0;
        float sum_up = 0.0;

        for (uint b = 0; b < batch_seq; b++) {
            uint cache_idx = b * hidden_dim + row;
            float gate = gate_cache[cache_idx];
            float up = up_cache[cache_idx];

            float grad_out = grad_output[cache_idx];
            float relu_up = max(0.0, up);
            float grad_gate = grad_out * relu_up;
            float grad_up = grad_out * gate * (up > 0.0 ? 1.0 : 0.0);

            uint x_idx = b * input_dim + col;
            sum_gate += grad_gate * input_data[x_idx];
            sum_up += grad_up * input_data[x_idx];
        }

        uint w_idx = row * input_dim + col;
        grad_W_gate[w_idx] = sum_gate;
        grad_W_up[w_idx] = sum_up;

    } else if (pass_type == 2) {
        // Compute gradient w.r.t. bias
        uint h = gl_GlobalInvocationID.x;
        if (h >= hidden_dim) return;

        float sum_gate = 0.0;
        float sum_up = 0.0;

        for (uint b = 0; b < batch_seq; b++) {
            uint cache_idx = b * hidden_dim + h;
            float gate = gate_cache[cache_idx];
            float up = up_cache[cache_idx];

            float grad_out = grad_output[cache_idx];
            float relu_up = max(0.0, up);
            float grad_gate = grad_out * relu_up;
            float grad_up = grad_out * gate * (up > 0.0 ? 1.0 : 0.0);

            sum_gate += grad_gate;
            sum_up += grad_up;
        }

        grad_b_gate[h] = sum_gate;
        grad_b_up[h] = sum_up;
    }
}
