#version 450

// Rational sigmoid: x / (1 + |x|)

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer Input {
    float input_data[];
};

layout(set = 0, binding = 1) buffer Output {
    float output_data[];
};

layout(push_constant) uniform PushConsts {
    uint total_elements;
};

void main() {
    uint gID = gl_GlobalInvocationID.x;
    if (gID >= total_elements) return;
    float x = input_data[gID];
    float ax = abs(x) + 1.0;
    output_data[gID] = x / ax;
}
