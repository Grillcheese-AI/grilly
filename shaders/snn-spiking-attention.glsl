#version 450

/*
 * Spiking Self-Attention (Spikformer-style)
 *
 * Computes spiking attention: out = (Q * K) * V
 * where Q, K, V are binary spike matrices.
 * No softmax — uses element-wise multiply.
 *
 * Bindings:
 *   0: Q          [batch * heads * seq_len * head_dim] (read)
 *   1: K          [batch * heads * seq_len * head_dim] (read)
 *   2: V          [batch * heads * seq_len * head_dim] (read)
 *   3: output     [batch * heads * seq_len * head_dim] (write)
 *
 * Push constants:
 *   batch_heads (uint) = batch_size * num_heads
 *   seq_len     (uint)
 *   head_dim    (uint)
 *   scale       (float) = 1.0 / sqrt(head_dim) or similar
 */

layout(local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0) readonly buffer QBuf   { float Q[]; };
layout(set = 0, binding = 1) readonly buffer KBuf   { float K[]; };
layout(set = 0, binding = 2) readonly buffer VBuf   { float V[]; };
layout(set = 0, binding = 3) writeonly buffer OutBuf { float out_buf[]; };

layout(push_constant) uniform PushConstants {
    uint  batch_heads;
    uint  seq_len;
    uint  head_dim;
    float scale;
};

void main() {
    uint bh = gl_GlobalInvocationID.z;
    uint row = gl_GlobalInvocationID.y;  // query position
    uint col = gl_GlobalInvocationID.x;  // head dim

    if (bh >= batch_heads || row >= seq_len || col >= head_dim) return;

    uint base_offset = bh * seq_len * head_dim;

    // Compute attention score: sum_d Q[row,d] * K[k,d] for all k
    // Then multiply by V
    float acc = 0.0;
    for (uint k = 0; k < seq_len; k++) {
        // QK attention: dot product of Q[row] and K[k]
        float qk = 0.0;
        for (uint d = 0; d < head_dim; d++) {
            qk += Q[base_offset + row * head_dim + d] * K[base_offset + k * head_dim + d];
        }
        qk *= scale;
        acc += qk * V[base_offset + k * head_dim + col];
    }

    out_buf[base_offset + row * head_dim + col] = acc;
}
