#version 450

// 1x1 Pointwise Convolution (DenseNet bottleneck / channel compression).
//
// Pure dot product per pixel across channel dimension.
// No spatial kernel — just matrix-vector multiply per (x, y).
// Used to compress accumulated DenseNet channels before next block.
//
// Dispatch: ((W+15)/16, (H+15)/16, out_channels)

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

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
    uint x = gl_GlobalInvocationID.x;
    uint y = gl_GlobalInvocationID.y;
    uint c_out = gl_GlobalInvocationID.z;

    if (x >= width || y >= height || c_out >= out_channels) return;

    float sum = 0.0;

    for (uint c_in = 0; c_in < in_channels; c_in++) {
        uint in_idx = (c_in * height + y) * width + x;
        uint w_idx = c_out * in_channels + c_in;
        sum += in_data[in_idx] * weights[w_idx];
    }

    uint out_idx = (c_out * height + y) * width + x;
    out_data[out_idx] = sum;
}
