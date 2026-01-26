#version 450

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Gradient from next layer
layout(set = 0, binding = 0) readonly buffer GradOutput {
    float grad_output[];
};

// Input to GELU
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

const float SQRT_2_OVER_PI = 0.7978845608028654;
const float COEFF = 0.044715;

void main() {
    uint gID = gl_GlobalInvocationID.x;
    
    if (gID >= total_elements) {
        return;
    }
    
    float x = input_data[gID];
    float grad_out = grad_output[gID];
    
    // GELU derivative
    // d/dx GELU(x) = 0.5 * (1 + tanh(z) + x * sech²(z) * dz/dx)
    // where z = sqrt(2/π) * (x + 0.044715 * x³)
    
    float x_cubed = x * x * x;
    float z = SQRT_2_OVER_PI * (x + COEFF * x_cubed);
    float tanh_z = tanh(z);
    float sech_z = 1.0 / cosh(z);
    float sech_sq = sech_z * sech_z;
    
    float dz_dx = SQRT_2_OVER_PI * (1.0 + 3.0 * COEFF * x * x);
    
    float gelu_grad = 0.5 * (1.0 + tanh_z + x * sech_sq * dz_dx);
    
    grad_input[gID] = grad_out * gelu_grad;
}
