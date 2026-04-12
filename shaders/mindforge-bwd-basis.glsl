#version 450

/*
 * MindForge Basis Mixer — Backward to Basis
 *
 * Computes the outer product contribution to every basis matrix:
 *     d_basis[i, j] += coeffs[i] * d_adapter[j]
 *
 * Embarrassingly parallel across the flattened (n_basis * adapter_size)
 * space. Coefficients are cached in shared memory, identical to the
 * forward shader.
 *
 * Caller MUST zero d_basis before the first dispatch. MindForge uses this
 * twice (once for d_A_basis, once for d_B_basis), so in-place += is the
 * contract.
 *
 * Dispatch: groupX = ceil(n_basis * adapter_size / 256)
 * Bindings:
 *   0: coeffs    (n_basis)                  readonly
 *   1: d_adapter (adapter_size)             readonly
 *   2: d_basis   (n_basis * adapter_size)   read/write
 * Push: (n_basis, adapter_size)
 */

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0) readonly buffer Coeffs    { float coeffs[]; };
layout(binding = 1) readonly buffer DAdapter  { float d_adapter[]; };
layout(binding = 2) buffer          DBasis    { float d_basis[]; };

layout(push_constant) uniform PushConstants {
    uint n_basis;
    uint adapter_size;
} params;

const uint MAX_BASIS = 256u;
shared float s_coeffs[MAX_BASIS];

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint idx = gl_GlobalInvocationID.x;

    if (tid < params.n_basis) {
        s_coeffs[tid] = coeffs[tid];
    }
    barrier();

    uint total_size = params.n_basis * params.adapter_size;
    if (idx >= total_size) return;

    uint basis_idx   = idx / params.adapter_size;
    uint adapter_idx = idx % params.adapter_size;

    float grad = s_coeffs[basis_idx] * d_adapter[adapter_idx];
    d_basis[idx] += grad;
}
