#version 450

// Algebraic softplus: 0.5 * (x + sqrt(x*x + c)) with c = 4/beta^2 — no exp/log.

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer Input {
    float input_data[];
};

layout(set = 0, binding = 1) buffer Output {
    float output_data[];
};

layout(push_constant) uniform PushConsts {
    uint total_elements;
    float c;  // 4 / (beta * beta), set from host
};

void main() {
    uint gID = gl_GlobalInvocationID.x;
    if (gID >= total_elements) return;
    float x = input_data[gID];
    float s = sqrt(x * x + c);
    output_data[gID] = 0.5 * (x + s);
}
