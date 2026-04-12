#version 450

/*
 * MindForge Basis Mixer — Forward
 *
 * Fuses the tensordot(coeffs, basis, axes=([0],[0])) operation used by
 * MindForge to forge LoRA A and B adapters from a shared trainable basis:
 *
 *     Adapter[j] = sum_i coeffs[i] * basis[i, j]
 *
 * Coefficients are tiny (typically n_basis=16) so we cache them in shared
 * memory and let every thread read them repeatedly without re-hitting VRAM.
 *
 * Dispatch: groupX = ceil(adapter_size / 256)
 * Bindings:
 *   0: coeffs     (n_basis)            readonly
 *   1: basis_data (n_basis * adapter_size) readonly, row-major per basis
 *   2: out_adapter (adapter_size)      writeonly
 * Push: (n_basis, adapter_size)
 */

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0) readonly buffer Coeffs { float coeffs[]; };
layout(binding = 1) readonly buffer Basis  { float basis_data[]; };
layout(binding = 2) writeonly buffer OutAdapter { float out_adapter[]; };

layout(push_constant) uniform PushConstants {
    uint n_basis;       // e.g. 16, must be <= MAX_BASIS
    uint adapter_size;  // flattened: rank * d_target
} params;

// Room for up to 256 basis pairs. Matches n_basis cap.
const uint MAX_BASIS = 256u;
shared float s_coeffs[MAX_BASIS];

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint idx = gl_GlobalInvocationID.x;

    // Cooperatively load the mixing coefficients into shared memory.
    if (tid < params.n_basis) {
        s_coeffs[tid] = coeffs[tid];
    }
    barrier();

    if (idx >= params.adapter_size) return;

    float acc = 0.0;
    for (uint i = 0u; i < params.n_basis; i++) {
        acc += s_coeffs[i] * basis_data[i * params.adapter_size + idx];
    }

    out_adapter[idx] = acc;
}
