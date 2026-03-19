#version 450
#extension GL_KHR_shader_subgroup_arithmetic : require

// 1x1 Conv Backward Input: dX = W^T @ dY per pixel.
// Subgroup-accelerated reduction over output channels.
// Dispatch: (width, height, in_channels)

layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

layout(std430, set = 0, binding = 0) readonly buffer GradOutputBuffer { float grad_out[]; };
layout(std430, set = 0, binding = 1) readonly buffer WeightBuffer { float weights[]; };
layout(std430, set = 0, binding = 2) writeonly buffer GradInputBuffer { float grad_in[]; };

layout(push_constant) uniform PushConstants {
    uint width;
    uint height;
    uint in_channels;
    uint out_channels;
};

void main() {
    uint px = gl_WorkGroupID.x;
    uint py = gl_WorkGroupID.y;
    uint c_in = gl_WorkGroupID.z;
    uint lane_id = gl_LocalInvocationID.x;

    if (px >= width || py >= height || c_in >= in_channels) return;

    float gradient_sum = 0.0;

    for (uint c_out_base = 0; c_out_base < out_channels; c_out_base += 32) {
        uint c_out = c_out_base + lane_id;
        float val = 0.0;

        if (c_out < out_channels) {
            uint grad_out_idx = (c_out * height + py) * width + px;
            uint w_idx = c_out * in_channels + c_in;
            val = grad_out[grad_out_idx] * weights[w_idx];
        }

        gradient_sum += subgroupAdd(val);
    }

    if (lane_id == 0) {
        uint grad_in_idx = (c_in * height + py) * width + px;
        grad_in[grad_in_idx] = gradient_sum;
    }
}
