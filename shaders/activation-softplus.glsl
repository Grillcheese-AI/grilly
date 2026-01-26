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
    float beta;  // Scaling parameter (default 1.0)
};

void main() {
    uint gID = gl_GlobalInvocationID.x;
    
    if (gID >= total_elements) {
        return;
    }
    
    float x = input_data[gID];
    
    // Softplus: log(1 + exp(β*x)) / β
    // For numerical stability:
    // - If x > threshold: softplus ≈ x
    // - Otherwise: use log1p for better precision
    
    float threshold = 20.0 / beta;
    float result;
    
    if (x * beta > threshold) {
        result = x;
    } else {
        // log1p(y) = log(1 + y) with better precision for small y
        result = log(1.0 + exp(beta * x)) / beta;
    }
    
    output_data[gID] = result;
}
