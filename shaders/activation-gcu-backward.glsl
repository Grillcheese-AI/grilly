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
};

void main() {
    uint gID = gl_GlobalInvocationID.x;

    if (gID >= total_elements) {
        return;
    }

    float x = input_data[gID];
    float grad_out = grad_output[gID];

    // GCU(x) = x * cos(x)
    // d/dx GCU(x) = cos(x) - x * sin(x)
    float gcu_derivative = cos(x) - x * sin(x);

    // Chain rule: grad_input = grad_output * local_derivative
    grad_input[gID] = grad_out * gcu_derivative;
}
