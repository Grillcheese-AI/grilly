#version 450

// Fused Residual Addition + ReLU for ResNet skip connections.
//
// Combines the output of a conv block with the skip connection
// input and applies ReLU in a single pass — zero intermediate VRAM write.
//
// out[i] = max(0, conv_data[i] + res_data[i])
//
// Dispatch: ((total_elements + 255) / 256, 1, 1)

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(std430, set = 0, binding = 0) readonly buffer ConvOutput {
    float conv_data[];
};

layout(std430, set = 0, binding = 1) readonly buffer ResidualInput {
    float res_data[];
};

layout(std430, set = 0, binding = 2) writeonly buffer OutputBuffer {
    float out_data[];
};

layout(push_constant) uniform PushConstants {
    uint total_elements;
};

void main() {
    uint idx = gl_GlobalInvocationID.x;

    if (idx >= total_elements) {
        return;
    }

    // Fused residual + ReLU in one pass
    float sum = conv_data[idx] + res_data[idx];
    out_data[idx] = max(0.0, sum);
}
