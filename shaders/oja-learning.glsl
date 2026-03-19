#version 450

// Oja's Rule for VSA pseudo-neuron plasticity.
//
// Each thread updates one hypervector: W[idx] adapts toward X[idx]
// using self-normalizing Oja update: ΔW = η·y·(X - y·W)
// where y = W·X is the VSA activation (similarity).
//
// Properties:
//   - Converges to ||W|| = 1.0 (self-normalizing)
//   - Extracts principal component of input stream
//   - Decays superposition noise, amplifies semantic signal
//
// Use for: codebook adaptation, hippocampal consolidation, cache sharpening.
//
// Reference: Oja, E. (1982). "Simplified neuron model as a principal
// component analyzer." J. Math. Biology, 15(3), 267-273.

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Binding 0: VSA Memory Vectors (pseudo-neurons) — read/write
layout(std430, set = 0, binding = 0) buffer MemoryBuffer {
    float W[];
};

// Binding 1: Incoming Experience/Perception Vectors — read only
layout(std430, set = 0, binding = 1) readonly buffer InputBuffer {
    float X[];
};

// Push constants for dynamic parameters
layout(push_constant) uniform PushConstants {
    uint num_vectors;  // Number of memories to update in this batch
    uint dim;          // VSA dimension (e.g., 512, 2048, 10240)
    float eta;         // Learning rate (e.g., 0.01)
};

void main() {
    uint idx = gl_GlobalInvocationID.x;

    if (idx >= num_vectors) return;

    uint offset = idx * dim;

    // ── Phase 1: Compute Activation y = W·X (VSA similarity) ──
    float y = 0.0;
    for (uint i = 0; i < dim; i++) {
        y += W[offset + i] * X[offset + i];
    }

    // ── Phase 2: Oja's Plasticity Update ──
    // ΔW = η·y·(X - y·W)
    // The (X - y·W) term is the Oja stabilizer that prevents explosion
    float eta_y = eta * y;

    for (uint i = 0; i < dim; i++) {
        float w_i = W[offset + i];
        float x_i = X[offset + i];
        W[offset + i] = w_i + eta_y * (x_i - y * w_i);
    }
}
