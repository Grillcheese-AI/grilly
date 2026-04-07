// Snippet: Addition-only linear (L1 distance)
// Computes -||w - x||_1 for a single (output, input) pair
// No multiplications — only add, sub, abs
float op_addition_linear(float w, float x) {
    return -abs(w - x);
}

// Snippet: Sign activation with threshold (ternary output)
float op_sign_activation(float x, float threshold) {
    return sign(x - threshold);
}

// Snippet: Additive receptance (sigmoid approximation)
// sigmoid(x) ≈ clamp(0.5 + 0.25 * x, 0, 1)
float op_additive_sigmoid(float x) {
    return clamp(0.5 + 0.25 * x, 0.0, 1.0);
}
