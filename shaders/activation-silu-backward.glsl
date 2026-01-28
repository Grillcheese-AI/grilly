#version 450

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Gradient from next layer
layout(set = 0, binding = 0) readonly buffer GradOutput {
    float grad_output[];
};

// Input to SiLU (pre-activation values)
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
};

void main() {
    uint gID = gl_GlobalInvocationID.x;

    if (gID >= total_elements) {
        return;
    }

    float x = input_data[gID];
    float grad_out = grad_output[gID];

    // SiLU(x) = x * sigmoid(x)
    // d/dx SiLU(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
    //              = sigmoid(x) * (1 + x * (1 - sigmoid(x)))

    float sigmoid_x = 1.0 / (1.0 + exp(-x));
    float silu_grad = sigmoid_x * (1.0 + x * (1.0 - sigmoid_x));

    grad_input[gID] = grad_out * silu_grad;
}
