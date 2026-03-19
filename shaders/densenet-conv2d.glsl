#version 450

// DenseNet 3x3 Conv with zero-copy concatenation.
//
// Reads from accumulated channels in a pre-allocated dense buffer,
// writes new features directly into the next channel slot.
// No memory copy, no separate concat operation.
//
// Dispatch: ((W+15)/16, (H+15)/16, growth_rate)

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(std430, set = 0, binding = 0) buffer DenseBuffer {
    float dense_data[];
};

layout(std430, set = 0, binding = 1) readonly buffer WeightBuffer {
    float weights[];
};

layout(push_constant) uniform PushConstants {
    uint width;
    uint height;
    uint current_in_channels;
    uint out_channels;
    uint out_channel_offset;
};

void main() {
    uint x = gl_GlobalInvocationID.x;
    uint y = gl_GlobalInvocationID.y;
    uint c_out = gl_GlobalInvocationID.z;

    if (x >= width || y >= height || c_out >= out_channels) return;

    float sum = 0.0;

    for (uint c_in = 0; c_in < current_in_channels; c_in++) {
        for (int ky = -1; ky <= 1; ky++) {
            for (int kx = -1; kx <= 1; kx++) {
                int ix = int(x) + kx;
                int iy = int(y) + ky;
                if (ix >= 0 && ix < int(width) && iy >= 0 && iy < int(height)) {
                    uint in_idx = (c_in * height + uint(iy)) * width + uint(ix);
                    uint w_idx = ((c_out * current_in_channels + c_in) * 3 + uint(ky + 1)) * 3 + uint(kx + 1);
                    sum += dense_data[in_idx] * weights[w_idx];
                }
            }
        }
    }

    // Fused ReLU
    float activated = max(0.0, sum);

    // Zero-copy concat: write to designated channel slot
    uint target_channel = out_channel_offset + c_out;
    uint out_idx = (target_channel * height + y) * width + x;
    dense_data[out_idx] = activated;
}
