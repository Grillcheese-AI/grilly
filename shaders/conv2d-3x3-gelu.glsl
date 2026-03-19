#version 450

// Fused 3x3 Conv2D + GELU for VSA-CNN perception frontend.
//
// One thread per output pixel. Z dimension maps to output channels.
// NCHW memory layout. Padding=1 (same spatial size).
// GELU fused at the end — no intermediate VRAM write.
//
// Dispatch: ((W+15)/16, (H+15)/16, out_channels)

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(std430, set = 0, binding = 0) readonly buffer InputBuffer {
    float in_data[];
};

layout(std430, set = 0, binding = 1) readonly buffer WeightBuffer {
    float weights[];
};

layout(std430, set = 0, binding = 2) readonly buffer BiasBuffer {
    float bias[];
};

layout(std430, set = 0, binding = 3) writeonly buffer OutputBuffer {
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

    if (x >= width || y >= height || c_out >= out_channels) {
        return;
    }

    float sum = bias[c_out];

    for (uint c_in = 0; c_in < in_channels; c_in++) {
        for (int ky = -1; ky <= 1; ky++) {
            for (int kx = -1; kx <= 1; kx++) {
                int ix = int(x) + kx;
                int iy = int(y) + ky;

                if (ix >= 0 && ix < int(width) && iy >= 0 && iy < int(height)) {
                    uint in_idx = (c_in * height + uint(iy)) * width + uint(ix);
                    uint w_idx = ((c_out * in_channels + c_in) * 3 + uint(ky + 1)) * 3 + uint(kx + 1);
                    sum += in_data[in_idx] * weights[w_idx];
                }
            }
        }
    }

    // Fused GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    float x3 = sum * sum * sum;
    float gelu = 0.5 * sum * (1.0 + tanh(0.7978845608 * (sum + 0.044715 * x3)));

    uint out_idx = (c_out * height + y) * width + x;
    out_data[out_idx] = gelu;
}
