#version 450

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Input 2D features (batch, dim)
layout(set = 0, binding = 0) readonly buffer Input2D {
    float input_2d[];
};

// Output 3D with time dimension (batch, 1, dim)
layout(set = 0, binding = 1) buffer Output3D {
    float output_3d[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint batch_size;
    uint dim;
};

void main() {
    uint gID = gl_GlobalInvocationID.x;
    uint total_elements = batch_size * dim;
    
    if (gID >= total_elements) {
        return;
    }
    
    // Simply copy and add time dimension of 1
    // Input: [batch, dim]
    // Output: [batch, 1, dim]
    output_3d[gID] = input_2d[gID];
}
