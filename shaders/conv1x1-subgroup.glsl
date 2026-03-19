#version 450

// Subgroup-accelerated 1x1 Pointwise Convolution.
//
// Uses VK_KHR_shader_subgroup_arithmetic for hardware-level reduction.
// 32 threads per workgroup cooperate to sum input channels via
// subgroupAdd() in a single clock cycle — no shared memory needed.
//
// One workgroup = one output pixel for one output channel.
// Dispatch: (width, height, out_channels)

#extension GL_KHR_shader_subgroup_arithmetic : require

layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

layout(std430, set = 0, binding = 0) readonly buffer InputBuffer {
    float in_data[];
};

layout(std430, set = 0, binding = 1) readonly buffer WeightBuffer {
    float weights[];
};

layout(std430, set = 0, binding = 2) writeonly buffer OutputBuffer {
    float out_data[];
};

layout(push_constant) uniform PushConstants {
    uint width;
    uint height;
    uint in_channels;
    uint out_channels;
};

void main() {
    uint px = gl_WorkGroupID.x;
    uint py = gl_WorkGroupID.y;
    uint c_out = gl_WorkGroupID.z;
    uint lane_id = gl_LocalInvocationID.x;

    if (px >= width || py >= height || c_out >= out_channels) return;

    float pixel_sum = 0.0;

    // Process input channels in chunks of 32 (subgroup width)
    for (uint c_in_base = 0; c_in_base < in_channels; c_in_base += 32) {
        uint c_in = c_in_base + lane_id;
        float val = 0.0;

        if (c_in < in_channels) {
            uint in_idx = (c_in * height + py) * width + px;
            uint w_idx = c_out * in_channels + c_in;
            val = in_data[in_idx] * weights[w_idx];
        }

        // Hardware reduction: sum across all 32 lanes in 1 cycle
        pixel_sum += subgroupAdd(val);
    }

    // Only lane 0 writes to avoid redundant VRAM writes
    if (lane_id == 0) {
        uint out_idx = (c_out * height + py) * width + px;
        out_data[out_idx] = pixel_sum;
    }
}
