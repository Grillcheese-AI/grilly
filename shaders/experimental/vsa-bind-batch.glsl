#version 450

// Bipolar binding (batch): element-wise multiplication across batches

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer VectorA {
    float a[];
};

layout(set = 0, binding = 1) readonly buffer VectorB {
    float b[];
};

layout(set = 0, binding = 2) buffer Result {
    float result[];
};

layout(push_constant) uniform PushConsts {
    uint total;
};

void main() {
    uint idx = gl_GlobalInvocationID.x;

    if (idx >= total) {
        return;
    }

    result[idx] = a[idx] * b[idx];
}
