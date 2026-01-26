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

// Sigmoid helper
float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

void main() {
    uint gID = gl_GlobalInvocationID.x;
    
    if (gID >= total_elements) {
        return;
    }
    
    float x = input_data[gID];
    
    // SiLU (Swish): x * sigmoid(x)
    float result = x * sigmoid(x);
    
    output_data[gID] = result;
}
