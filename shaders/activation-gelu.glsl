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
};

const float SQRT_2_OVER_PI = 0.7978845608028654;  // sqrt(2/π)
const float COEFF = 0.044715;

void main() {
    uint gID = gl_GlobalInvocationID.x;
    
    if (gID >= total_elements) {
        return;
    }
    
    float x = input_data[gID];
    
    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    // Clamp x to prevent overflow in x³ computation
    float x_clamped = clamp(x, -50.0, 50.0);
    float x_cubed = x_clamped * x_clamped * x_clamped;
    
    // Compute inner term with overflow protection
    float inner_term = x_clamped + COEFF * x_cubed;
    float inner = SQRT_2_OVER_PI * inner_term;
    
    // tanh is bounded, so this should be safe
    float tanh_val = tanh(inner);
    float gelu = 0.5 * x * (1.0 + tanh_val);
    
    // Handle edge cases: if input was clamped, ensure output is reasonable
    if (abs(x) > 50.0) {
        // For very large positive values, GELU approaches x
        // For very large negative values, GELU approaches 0
        gelu = (x > 0.0) ? x : 0.0;
    }
    
    output_data[gID] = gelu;
}
