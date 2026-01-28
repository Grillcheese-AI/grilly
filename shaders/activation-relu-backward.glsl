#version 450

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Gradient from next layer
layout(set = 0, binding = 0) readonly buffer GradOutput {
    float grad_output[];
};

// Input to ReLU (pre-activation values)
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

    // ReLU derivative: 1 if x > 0, else 0
    float relu_grad = (x > 0.0) ? 1.0 : 0.0;

    grad_input[gID] = grad_out * relu_grad;
}
