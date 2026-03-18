#version 450

// Sanger's Generalized Hebbian Algorithm (GHA) — Online PCA
//
// Streaming O(d) principal component extraction for neurogenesis-driven
// dimension expansion. When a hypervector bundle reaches 75% capacity,
// this shader whitens the data by extracting orthogonal principal
// components, then new dimensions are spawned from the residual.
//
// Each invocation updates one weight vector w_j for one component j.
// The algorithm extracts principal components sequentially and
// orthogonally via Gram-Schmidt-like subtraction in the update rule.
//
// Sanger's rule for component j:
//   y_j = w_j · x
//   w_j += lr * (y_j * x - y_j * sum_{k<=j} y_k * w_k)
//
// This finds the top-k principal components in a single streaming pass.

layout(local_size_x = 256) in;

// Weight matrix: (num_components, dim) — each row is a principal component
layout(set=0, binding=0) buffer Weights { float W[]; };

// Input vector: (dim,) — current data sample
layout(set=0, binding=1) readonly buffer Input { float x[]; };

// Output projections: (num_components,) — y_j = w_j · x
layout(set=0, binding=2) buffer Projections { float y[]; };

layout(push_constant) uniform Params {
    uint dim;              // vector dimension
    uint num_components;   // number of PCA components to extract
    float learning_rate;   // Hebbian learning rate (e.g., 0.01)
    uint pass_type;        // 0 = compute projections, 1 = update weights
};

void main() {
    uint idx = gl_GlobalInvocationID.x;

    if (pass_type == 0u) {
        // Pass 0: Compute projections y_j = w_j · x for each component j
        if (idx >= num_components) return;

        float dot = 0.0;
        uint base = idx * dim;
        for (uint d = 0u; d < dim; d++) {
            dot += W[base + d] * x[d];
        }
        y[idx] = dot;

    } else {
        // Pass 1: Update weights using Sanger's rule
        // Each thread updates one element of the weight matrix
        if (idx >= num_components * dim) return;

        uint j = idx / dim;  // which component
        uint d = idx % dim;  // which dimension

        float y_j = y[j];

        // Compute the Gram-Schmidt correction sum: sum_{k<=j} y_k * W[k,d]
        float correction = 0.0;
        for (uint k = 0u; k <= j; k++) {
            correction += y[k] * W[k * dim + d];
        }

        // Sanger's update: w_j,d += lr * y_j * (x_d - correction)
        W[idx] += learning_rate * y_j * (x[d] - correction);
    }
}
