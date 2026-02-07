#version 450

// Bundling (batch): superposition with majority voting across batches

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer Vectors {
    float vectors[];  // (batch * num_vectors * dim)
};

layout(set = 0, binding = 1) buffer Result {
    float result[];   // (batch * dim)
};

layout(push_constant) uniform PushConsts {
    uint dim;
    uint num_vectors;
    uint batch;
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint total = dim * batch;

    if (idx >= total) {
        return;
    }

    uint b = idx / dim;
    uint d = idx - b * dim;

    float sum = 0.0;
    uint base = b * num_vectors * dim;
    for (uint i = 0; i < num_vectors; i++) {
        sum += vectors[base + i * dim + d];
    }

    result[idx] = sign(sum + 1e-8);
}
