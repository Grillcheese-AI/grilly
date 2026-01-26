#version 450

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Input A (residual path)
layout(set = 0, binding = 0) readonly buffer InputA {
    float a[];
};

// Input B (main path)
layout(set = 0, binding = 1) readonly buffer InputB {
    float b[];
};

// Output (A + B)
layout(set = 0, binding = 2) buffer Output {
    float output_data[];
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
    
    // Simple element-wise addition for residual connection
    output_data[gID] = a[gID] + b[gID];
}
