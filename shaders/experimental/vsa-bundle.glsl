#version 450

// Bundling: superposition with majority voting
// Sums vectors and applies sign function for bipolar output

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer Vectors {
    float vectors[];  // (num_vectors * dim)
};

layout(set = 0, binding = 1) buffer Result {
    float result[];
};

layout(push_constant) uniform PushConsts {
    uint dim;
    uint num_vectors;
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    
    if (idx >= dim) {
        return;
    }
    
    // Sum across all vectors at this dimension
    float sum = 0.0;
    for (uint i = 0; i < num_vectors; i++) {
        sum += vectors[i * dim + idx];
    }
    
    // Majority vote: sign of sum (with small epsilon to break ties)
    result[idx] = sign(sum + 1e-8);
}
