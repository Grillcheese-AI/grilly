#version 450

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Gradient from next layer
layout(set = 0, binding = 0) readonly buffer GradOutput {
    float grad_output[];
};

// tanh output (cached from forward pass)
layout(set = 0, binding = 1) readonly buffer TanhOutput {
    float tanh_out[];
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

    // d(tanh)/dx = 1 - tanh(x)^2
    float t = tanh_out[gID];
    grad_input[gID] = grad_output[gID] * (1.0 - t * t);
}
