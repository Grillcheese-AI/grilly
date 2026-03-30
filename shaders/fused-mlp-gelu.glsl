#version 450
#extension GL_KHR_shader_subgroup_arithmetic : require

// Fused MLP: Linear(d_in→d_hidden) → GELU → Linear(d_hidden→d_out)
//
// Eliminates two VRAM round-trips by keeping the intermediate activations
// in registers/LDS between fc1 and fc2. For a transformer MLP
// (768→3072→768), this saves writing and reading 3072 floats per token.
//
// Each workgroup processes one row (one token) of the input matrix.
// Threads within the workgroup cooperatively compute fc1, apply GELU,
// then compute fc2 — all without touching global memory for intermediates.
//
// Dispatch: workgroups_x = seq_len (one per token)

layout(constant_id = 0) const uint D_IN = 768;
layout(constant_id = 1) const uint D_HIDDEN = 3072;
layout(constant_id = 2) const uint D_OUT = 768;

layout(local_size_x = 256) in;

layout(std430, set = 0, binding = 0) readonly buffer Input   { float input_data[]; };
layout(std430, set = 0, binding = 1) readonly buffer W1      { float w1[]; };      // (d_hidden, d_in)
layout(std430, set = 0, binding = 2) readonly buffer B1      { float b1[]; };      // (d_hidden,)
layout(std430, set = 0, binding = 3) readonly buffer W2      { float w2[]; };      // (d_out, d_hidden)
layout(std430, set = 0, binding = 4) readonly buffer B2      { float b2[]; };      // (d_out,)
layout(std430, set = 0, binding = 5) writeonly buffer Output { float output_data[]; };

const float SQRT_2_OVER_PI = 0.7978845608028654;
const float COEFF = 0.044715;

// Shared memory for intermediate hidden activations (after GELU)
shared float shared_hidden[3072];  // Max d_hidden

void main() {
    uint token_idx = gl_WorkGroupID.x;
    uint tid = gl_LocalInvocationID.x;
    uint input_offset = token_idx * D_IN;
    uint output_offset = token_idx * D_OUT;

    // --- STEP 1: fc1 + GELU (d_in → d_hidden, with activation) ---
    // Each thread computes one or more hidden units
    for (uint h = tid; h < D_HIDDEN; h += gl_WorkGroupSize.x) {
        // Dot product: w1[h] . input + b1[h]
        float acc = b1[h];
        for (uint j = 0; j < D_IN; j++) {
            acc += w1[h * D_IN + j] * input_data[input_offset + j];
        }

        // Fused GELU with overflow protection
        float x = acc;
        float gelu_val;
        if (x > 10.0) {
            gelu_val = x;
        } else if (x < -10.0) {
            gelu_val = 0.0;
        } else {
            float x3 = x * x * x;
            float inner = SQRT_2_OVER_PI * (x + COEFF * x3);
            gelu_val = 0.5 * x * (1.0 + tanh(inner));
        }

        // Store in shared memory (stays in LDS, never hits VRAM)
        shared_hidden[h] = gelu_val;
    }

    // Sync: all hidden units must be computed before fc2
    barrier();

    // --- STEP 2: fc2 (d_hidden → d_out) ---
    for (uint o = tid; o < D_OUT; o += gl_WorkGroupSize.x) {
        float acc = b2[o];
        for (uint h = 0; h < D_HIDDEN; h++) {
            acc += w2[o * D_HIDDEN + h] * shared_hidden[h];
        }
        output_data[output_offset + o] = acc;
    }
}
