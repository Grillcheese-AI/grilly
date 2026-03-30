// Snippet: GELU activation (exact tanh approximation)
// Safe for |x| > 10 (early exit prevents exp overflow)
float op_gelu(float x) {
    if (x > 10.0) return x;
    if (x < -10.0) return 0.0;
    float c = 0.7978845608; // sqrt(2/pi)
    return 0.5 * x * (1.0 + tanh(c * (x + 0.044715 * x * x * x)));
}
