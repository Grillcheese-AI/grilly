#version 450

// Adaptive Average Pool to fixed 3x3 output for VSA spatial grid.
//
// Takes any spatial feature map (e.g., 10x10) and pools it into
// exactly 9 spatial vectors — one per grid position. This is the
// "lens" that bridges CNN features to per-position VSA block-codes.
//
// Thread mapping: X=3, Y=3 (fixed output), Z=channels (batched).
// Dispatch: (1, 1, (channels+15)/16)

layout(local_size_x = 3, local_size_y = 3, local_size_z = 16) in;

layout(std430, set = 0, binding = 0) readonly buffer InputBuffer {
    float in_data[];
};

layout(std430, set = 0, binding = 1) writeonly buffer OutputBuffer {
    float out_data[];
};

layout(push_constant) uniform PushConstants {
    uint in_width;
    uint in_height;
    uint channels;
};

void main() {
    uint ox = gl_GlobalInvocationID.x;  // 0..2
    uint oy = gl_GlobalInvocationID.y;  // 0..2
    uint c  = gl_GlobalInvocationID.z;  // channel

    if (ox >= 3 || oy >= 3 || c >= channels) {
        return;
    }

    // Adaptive binning: PyTorch-style floor/ceil division
    uint start_x = (ox * in_width) / 3;
    uint end_x   = ((ox + 1) * in_width + 2) / 3;
    uint start_y = (oy * in_height) / 3;
    uint end_y   = ((oy + 1) * in_height + 2) / 3;

    if (end_x > in_width)  end_x = in_width;
    if (end_y > in_height) end_y = in_height;

    float sum = 0.0;
    float count = 0.0;

    for (uint iy = start_y; iy < end_y; iy++) {
        for (uint ix = start_x; ix < end_x; ix++) {
            uint in_idx = (c * in_height + iy) * in_width + ix;
            sum += in_data[in_idx];
            count += 1.0;
        }
    }

    uint out_idx = (c * 3 + oy) * 3 + ox;
    out_data[out_idx] = sum / max(count, 1.0);
}
