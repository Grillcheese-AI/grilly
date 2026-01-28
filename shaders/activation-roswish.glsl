#version 450

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Input tensor
layout(set = 0, binding = 0) readonly buffer Input {
    float input_data[];
};

// Output tensor
layout(set = 0, binding = 1) buffer Output {
    float output_data[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint total_elements;
    float alpha;  // Learnable rotation parameter
    float beta;   // Learnable gating parameter
};

void main() {
    uint gID = gl_GlobalInvocationID.x;

    if (gID >= total_elements) {
        return;
    }

    float x = input_data[gID];

    // RoSwish: (x + α) * sigmoid(β * x) - 0.5 * α
    // Numerically stable sigmoid computation
    float beta_x = beta * x;
    float sigmoid_bx;

    if (beta_x >= 0.0) {
        // Positive: sigmoid(z) = 1 / (1 + exp(-z))
        sigmoid_bx = 1.0 / (1.0 + exp(-beta_x));
    } else {
        // Negative: sigmoid(z) = exp(z) / (1 + exp(z))
        float exp_bx = exp(beta_x);
        sigmoid_bx = exp_bx / (1.0 + exp_bx);
    }

    output_data[gID] = (x + alpha) * sigmoid_bx - 0.5 * alpha;
}
