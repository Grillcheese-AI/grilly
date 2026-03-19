#version 450

// 2x2 Max Pooling with stride 2 for VSA-CNN.
//
// One thread per output pixel. Z maps to channels.
// Halves spatial dimensions, preserves channel count.
//
// Dispatch: ((out_W+15)/16, (out_H+15)/16, channels)

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(std430, set = 0, binding = 0) readonly buffer InputBuffer {
    float in_data[];
};

layout(std430, set = 0, binding = 1) writeonly buffer OutputBuffer {
    float out_data[];
};

layout(push_constant) uniform PushConstants {
    uint out_width;
    uint out_height;
    uint channels;
    uint in_width;
};

void main() {
    uint ox = gl_GlobalInvocationID.x;
    uint oy = gl_GlobalInvocationID.y;
    uint c = gl_GlobalInvocationID.z;

    if (ox >= out_width || oy >= out_height || c >= channels) {
        return;
    }

    uint ix = ox * 2;
    uint iy = oy * 2;
    uint in_height = out_height * 2;

    float max_val = -1e30;

    for (uint dy = 0; dy < 2; dy++) {
        for (uint dx = 0; dx < 2; dx++) {
            uint in_idx = (c * in_height + (iy + dy)) * in_width + (ix + dx);
            max_val = max(max_val, in_data[in_idx]);
        }
    }

    uint out_idx = (c * out_height + oy) * out_width + ox;
    out_data[out_idx] = max_val;
}
