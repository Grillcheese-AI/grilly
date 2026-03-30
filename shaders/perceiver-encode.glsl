/// perceiver-encode.glsl — Perceiver IO cross-attention (encode step).
///
/// Register-pinned Q with streaming K/V and online softmax.
/// Zero LDS, zero barriers. 1 thread = 1 latent token.
///
/// Architecture-optimal for RDNA2 (RX 6750 XT):
///   - D=64 → 16 vec4 registers per thread (64 VGPRs of 256 available)
///   - Wavefront of 32 threads streams K/V with perfect coalescing
///   - Online softmax: O(1) VRAM regardless of input length M
///   - vec4 loads: 128-bit coalesced reads, dot() → v_dot4_f32_f32
///
/// Dispatch: workgroups_x = ceil(N / 64), 1, 1
/// Push constants: seq_M, seq_N, head_dim, scale

#version 450
#extension GL_KHR_shader_subgroup_arithmetic : require

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

// Vectorized 128-bit global memory bindings
layout(std430, set = 0, binding = 0) readonly buffer QBuf  { vec4 Q_in[];  };
layout(std430, set = 0, binding = 1) readonly buffer KBuf  { vec4 K_in[];  };
layout(std430, set = 0, binding = 2) readonly buffer VBuf  { vec4 V_in[];  };
layout(std430, set = 0, binding = 3) writeonly buffer OutBuf { vec4 Out[];  };

layout(push_constant) uniform Params {
    uint seq_M;      // Input sequence length (can be massive)
    uint seq_N;      // Latent sequence length (e.g., 256)
    uint head_dim;   // D per head (e.g., 64)
    float scale;     // 1.0 / sqrt(D)
} p;

void main() {
    uint latent_idx = gl_GlobalInvocationID.x;
    if (latent_idx >= p.seq_N) return;

    // D/4 because we fetch vec4 (4 floats at a time)
    uint vec4_dim = p.head_dim / 4u;

    // 1. PIN QUERY TO HARDWARE REGISTERS (VGPRs)
    // D=64 → 16 vec4 registers. RDNA2 has 256 VGPRs per thread.
    vec4 q_reg[16];

    for (uint d = 0; d < vec4_dim; d++) {
        q_reg[d] = Q_in[latent_idx * vec4_dim + d];
    }

    // 2. ONLINE SOFTMAX REGISTERS (Flash Attention math)
    float m_i = -1e30;  // Running max tracker
    float l_i = 0.0;    // Running sum-of-exps tracker
    vec4 out_acc[16];    // Output accumulator (pinned in VGPRs)

    for (uint d = 0; d < 16; d++) {
        out_acc[d] = vec4(0.0);
    }

    // 3. STREAM THE MASSIVE INPUT SEQUENCE (K and V)
    // Single pass over M. O(1) VRAM usage via online softmax.
    for (uint m = 0; m < p.seq_M; m++) {

        // --- Compute dot product (Q · K^T) ---
        float score = 0.0;
        uint k_base = m * vec4_dim;

        for (uint d = 0; d < vec4_dim; d++) {
            vec4 k_val = K_in[k_base + d];
            score += dot(q_reg[d], k_val);  // Hardware v_dot4_f32_f32
        }
        score *= p.scale;

        // --- Online softmax update ---
        float m_old = m_i;
        m_i = max(m_i, score);

        float exp_val = exp(score - m_i);
        float exp_old = exp(m_old - m_i);

        l_i = l_i * exp_old + exp_val;

        // --- Accumulate V (rescale previous by exp_old, add new) ---
        uint v_base = m * vec4_dim;

        for (uint d = 0; d < vec4_dim; d++) {
            out_acc[d] = out_acc[d] * exp_old + V_in[v_base + d] * exp_val;
        }
    }

    // 4. NORMALIZE AND WRITE BACK
    float inv_l_i = 1.0 / l_i;
    uint out_base = latent_idx * vec4_dim;

    for (uint d = 0; d < vec4_dim; d++) {
        Out[out_base + d] = out_acc[d] * inv_l_i;
    }
}
