/// perceiver-encode.glsl — Perceiver IO cross-attention (encode step).
///
/// Register-pinned Q with streaming K/V and online softmax.
/// Zero LDS, zero barriers. 1 thread = 1 latent token.
///
/// Multi-head support: stride + head_offset push constants allow reading
/// head slices from full (seq, D_model) buffers without de-interleaving.
/// Each head dispatch reads columns [head_offset, head_offset+head_dim).
///
/// Architecture-optimal for RDNA2 (RX 6750 XT):
///   - D=64 → 16 vec4 registers per thread (64 VGPRs of 256 available)
///   - Full occupancy: 4+ wavefronts per SIMD at D=64
///   - Online softmax: O(1) VRAM regardless of input length M
///   - vec4 loads: 128-bit coalesced reads, dot() → v_dot4_f32_f32
///
/// Dispatch: workgroups_x = ceil(N / 64), 1, 1
/// Push constants: seq_M, seq_N, head_dim, scale, stride, head_offset

#version 450
#extension GL_KHR_shader_subgroup_arithmetic : require

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

// Buffers store full (seq, D_model) row-major. We read head slices via stride.
layout(std430, set = 0, binding = 0) readonly buffer QBuf   { float Q_in[];  };
layout(std430, set = 0, binding = 1) readonly buffer KBuf   { float K_in[];  };
layout(std430, set = 0, binding = 2) readonly buffer VBuf   { float V_in[];  };
layout(std430, set = 0, binding = 3) writeonly buffer OutBuf { float Out[];   };

layout(push_constant) uniform Params {
    uint seq_M;         // Input sequence length
    uint seq_N;         // Latent sequence length
    uint head_dim;      // D per head (e.g., 64). Must be multiple of 4.
    float scale;        // 1.0 / sqrt(head_dim)
    uint stride;        // Row stride in floats (= D_model for multi-head, = head_dim for single-head)
    uint head_offset;   // Column offset in floats (= h * head_dim)
} p;

void main() {
    uint latent_idx = gl_GlobalInvocationID.x;
    if (latent_idx >= p.seq_N) return;

    uint hd = p.head_dim;
    uint hd4 = hd / 4u;  // vec4 count per head

    // 1. PIN QUERY TO VGPRs
    // D=64 → 16 vec4 registers. Reads from Q_in at row=latent_idx, cols=[head_offset, head_offset+hd)
    float q_reg[64];  // Max head_dim=64

    uint q_row = latent_idx * p.stride + p.head_offset;
    for (uint d = 0; d < hd; d++) {
        q_reg[d] = Q_in[q_row + d];
    }

    // 2. ONLINE SOFTMAX
    float m_i = -1e30;
    float l_i = 0.0;
    float out_acc[64];
    for (uint d = 0; d < hd; d++) {
        out_acc[d] = 0.0;
    }

    // 3. STREAM K/V
    for (uint m = 0; m < p.seq_M; m++) {
        uint kv_row = m * p.stride + p.head_offset;

        // Dot product Q · K
        float score = 0.0;
        for (uint d = 0; d < hd; d += 4) {
            score += q_reg[d]   * K_in[kv_row + d]
                   + q_reg[d+1] * K_in[kv_row + d+1]
                   + q_reg[d+2] * K_in[kv_row + d+2]
                   + q_reg[d+3] * K_in[kv_row + d+3];
        }
        score *= p.scale;

        // Online softmax update
        float m_old = m_i;
        m_i = max(m_i, score);
        float exp_val = exp(score - m_i);
        float exp_old = exp(m_old - m_i);
        l_i = l_i * exp_old + exp_val;

        // Accumulate V
        for (uint d = 0; d < hd; d += 4) {
            out_acc[d]   = out_acc[d]   * exp_old + V_in[kv_row + d]   * exp_val;
            out_acc[d+1] = out_acc[d+1] * exp_old + V_in[kv_row + d+1] * exp_val;
            out_acc[d+2] = out_acc[d+2] * exp_old + V_in[kv_row + d+2] * exp_val;
            out_acc[d+3] = out_acc[d+3] * exp_old + V_in[kv_row + d+3] * exp_val;
        }
    }

    // 4. NORMALIZE AND WRITE
    float inv_l_i = 1.0 / l_i;
    uint out_row = latent_idx * p.stride + p.head_offset;
    for (uint d = 0; d < hd; d++) {
        Out[out_row + d] = out_acc[d] * inv_l_i;
    }
}
