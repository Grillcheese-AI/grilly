#version 450

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Gradient from next layer
layout(set = 0, binding = 0) readonly buffer GradOutput {
    float grad_output[];
};

// Input tensor (for computing derivative)
layout(set = 0, binding = 1) readonly buffer Input {
    float input_data[];
};

// Gradient w.r.t. input
layout(set = 0, binding = 2) buffer GradInput {
    float grad_input[];
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
    float grad_out = grad_output[gID];

    // RoSwish(x) = (x + α) * sigmoid(β * x) - 0.5 * α
    // d/dx RoSwish = sigmoid(β*x) + β*(x + α)*sigmoid(β*x)*(1 - sigmoid(β*x))

    // Compute sigmoid(β*x) numerically stable
    float beta_x = beta * x;
    float sigmoid_bx;

    if (beta_x >= 0.0) {
        sigmoid_bx = 1.0 / (1.0 + exp(-beta_x));
    } else {
        float exp_bx = exp(beta_x);
        sigmoid_bx = exp_bx / (1.0 + exp_bx);
    }

    // Derivative: sigmoid + β*(x + α)*sigmoid*(1 - sigmoid)
    float roswish_derivative = sigmoid_bx + beta * (x + alpha) * sigmoid_bx * (1.0 - sigmoid_bx);

    // Chain rule: grad_input = grad_output * local_derivative
    grad_input[gID] = grad_out * roswish_derivative;
}
