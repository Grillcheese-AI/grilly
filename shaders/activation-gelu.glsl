#version 450

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Input tensor
layout(set = 0, binding = 0) readonly buffer Input {
    float input_data[];
};

// Output tensor
layout(set = 0, binding = 1) buffer Output {
    float output_data[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint total_elements;
};

const float SQRT_2_OVER_PI = 0.7978845608028654;  // sqrt(2/π)
const float COEFF = 0.044715;

void main() {
    uint gID = gl_GlobalInvocationID.x;
    
    if (gID >= total_elements) {
        return;
    }
    
    float x = input_data[gID];

    // GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    //
    // Overflow analysis: the tanh argument grows as O(x³).
    // tanh(t) uses exp(2t), which overflows float32 at t ≈ 44.
    // SQRT_2_OVER_PI * (x + 0.044715 * x³) > 44 when |x| > ~4.8
    // But tanh saturates to ±1 well before that, so the real danger is
    // x³ overflowing float32 (max ~3.4e38): x³ overflows at |x| > 3200.
    // However, exp(2*inner) overflows at inner > 88, and inner > 88
    // happens at |x| > ~10. On some GPU architectures (RDNA 2, RDNA 3),
    // tanh() returns NaN instead of ±1 when the argument is too large.
    //
    // Fix: for |x| > 10, GELU is within 1e-7 of its asymptotic form:
    //   x > 10  → GELU(x) ≈ x  (tanh → 1)
    //   x < -10 → GELU(x) ≈ 0  (tanh → -1, 1+(-1) = 0)
    // No precision loss, just skip the tanh computation entirely.

    float gelu;
    if (x > 10.0) {
        gelu = x;  // GELU → x for large positive
    } else if (x < -10.0) {
        gelu = 0.0;  // GELU → 0 for large negative
    } else {
        float x_cubed = x * x * x;
        float inner = SQRT_2_OVER_PI * (x + COEFF * x_cubed);
        gelu = 0.5 * x * (1.0 + tanh(inner));
    }

    output_data[gID] = gelu;
}
