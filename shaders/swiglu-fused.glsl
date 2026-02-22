#version 450

// Fused SwiGLU: output = SiLU(input @ gate_proj.T) * (input @ up_proj.T)
// Eliminates 2 intermediate buffers vs separate matmul + activation

layout (local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

// Input: (batch_seq, input_dim)
layout(set = 0, binding = 0) readonly buffer Input {
    float input_data[];
};

// Gate projection weights: (intermediate_size, input_dim)
layout(set = 0, binding = 1) readonly buffer GateWeights {
    float gate_W[];
};

// Up projection weights: (intermediate_size, input_dim)
layout(set = 0, binding = 2) readonly buffer UpWeights {
    float up_W[];
};

// Output: (batch_seq, intermediate_size)
layout(set = 0, binding = 3) buffer Output {
    float output_data[];
};

layout(push_constant) uniform PushConsts {
    uint batch_seq;
    uint input_dim;
    uint intermediate_size;
};

// Numerically stable sigmoid
float stable_sigmoid(float x) {
    if (x >= 0.0) {
        return 1.0 / (1.0 + exp(-x));
    } else {
        float ex = exp(x);
        return ex / (1.0 + ex);
    }
}

void main() {
    uint col = gl_GlobalInvocationID.x;  // intermediate_size index
    uint row = gl_GlobalInvocationID.y;  // batch_seq index

    if (row >= batch_seq || col >= intermediate_size) return;

    // Compute gate = input @ gate_proj.T and up = input @ up_proj.T simultaneously
    float gate_sum = 0.0;
    float up_sum = 0.0;

    for (uint k = 0; k < input_dim; k++) {
        float x = input_data[row * input_dim + k];
        gate_sum += x * gate_W[col * input_dim + k];
        up_sum += x * up_W[col * input_dim + k];
    }

    // SiLU(gate) = gate * sigmoid(gate)
    float silu_gate = gate_sum * stable_sigmoid(gate_sum);

    // SwiGLU = SiLU(gate) * up
    output_data[row * intermediate_size + col] = silu_gate * up_sum;
}
