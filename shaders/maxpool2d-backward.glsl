#version 450
#extension GL_EXT_shader_atomic_float : enable

/*
 * MaxPool2d Backward Pass
 *
 * Routes gradients to max positions using saved indices.
 */

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0) readonly buffer GradOutputBuffer {
    float grad_output[];
};

layout(binding = 1) readonly buffer IndicesBuffer {
    uint indices[];  // Max indices from forward pass
};

layout(binding = 2) buffer GradInputBuffer {
    float grad_input[];
};

layout(push_constant) uniform PushConstants {
    uint output_size;
    uint input_size;
} params;

void main() {
    uint idx = gl_GlobalInvocationID.x;

    if (idx >= params.output_size) {
        return;
    }

    // Route gradient to max position
    uint max_idx = indices[idx];
    atomicAdd(grad_input[max_idx], grad_output[idx]);
}
