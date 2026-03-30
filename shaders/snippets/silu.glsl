// Snippet: SiLU (Swish) activation
float op_silu(float x) {
    return x / (1.0 + exp(-clamp(x, -20.0, 20.0)));
}
