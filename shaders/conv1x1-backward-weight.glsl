#version 450
#extension GL_EXT_shader_atomic_float : require

// 1x1 Conv Backward Weight: dW = sum_{x,y} dY[c_out,y,x] * X[c_in,y,x]
// Atomic float addition to accumulate across spatial dimensions.
// Dispatch: ((W+15)/16, (H+15)/16, 1)

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(std430, set = 0, binding = 0) readonly buffer InputBuffer { float in_data[]; };
layout(std430, set = 0, binding = 1) readonly buffer GradOutputBuffer { float grad_out[]; };
layout(std430, set = 0, binding = 2) buffer GradWeightBuffer { float grad_weights[]; };

layout(push_constant) uniform PushConstants {
    uint width;
    uint height;
    uint in_channels;
    uint out_channels;
};

void main() {
    uint px = gl_GlobalInvocationID.x;
    uint py = gl_GlobalInvocationID.y;

    if (px >= width || py >= height) return;

    for (uint c_out = 0; c_out < out_channels; c_out++) {
        uint grad_out_idx = (c_out * height + py) * width + px;
        float dy = grad_out[grad_out_idx];

        if (dy == 0.0) continue;

        for (uint c_in = 0; c_in < in_channels; c_in++) {
            uint in_idx = (c_in * height + py) * width + px;
            float x = in_data[in_idx];

            uint w_idx = c_out * in_channels + c_in;
            atomicAdd(grad_weights[w_idx], dy * x);
        }
    }
}
