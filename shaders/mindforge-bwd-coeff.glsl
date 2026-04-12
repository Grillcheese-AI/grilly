#version 450
#extension GL_KHR_shader_subgroup_arithmetic : require

/*
 * MindForge Basis Mixer — Backward to Coefficients
 *
 * Computes:
 *     d_coeffs[i] += sum_j d_adapter[j] * basis[i, j]
 *
 * One workgroup per basis index (groupX = n_basis). Each workgroup sweeps
 * its entire basis slice in a grid-stride loop, reduces with subgroupAdd,
 * and accumulates the single scalar into d_coeffs[basis_idx].
 *
 * Caller MUST zero d_coeffs before the first dispatch. MindForge uses this
 * twice (once for d_A, once for d_B), so in-place += is the contract.
 *
 * Bindings:
 *   0: d_adapter (adapter_size)             readonly
 *   1: basis     (n_basis * adapter_size)   readonly
 *   2: d_coeffs  (n_basis)                  read/write — NOT writeonly,
 *                                            because we accumulate in place
 * Push: (adapter_size)
 */

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0) readonly buffer DAdapter { float d_adapter[]; };
layout(binding = 1) readonly buffer Basis    { float basis_data[]; };
layout(binding = 2) buffer DCoeffs           { float d_coeffs[]; };

layout(push_constant) uniform PushConstants {
    uint adapter_size;
} params;

// Max gl_NumSubgroups for 256 threads on typical GPUs is ceil(256/sgSize)
// = 8 (32-wide), 4 (64-wide), 2 (128-wide). 16 is safely oversized.
shared float wg_sum[16];

void main() {
    uint basis_idx = gl_WorkGroupID.x;       // which coefficient we compute
    uint tid       = gl_LocalInvocationID.x;
    uint sg_id     = gl_SubgroupID;

    // Grid-stride dot product over this basis slice.
    uint basis_offset = basis_idx * params.adapter_size;
    float local_sum = 0.0;
    for (uint i = tid; i < params.adapter_size; i += 256u) {
        local_sum += d_adapter[i] * basis_data[basis_offset + i];
    }

    // Subgroup-level reduction.
    float sg_sum = subgroupAdd(local_sum);
    if (subgroupElect()) {
        wg_sum[sg_id] = sg_sum;
    }
    barrier();

    // Cross-subgroup reduction on thread 0, then accumulate into d_coeffs.
    if (tid == 0u) {
        float final_sum = 0.0;
        for (uint i = 0u; i < gl_NumSubgroups; i++) {
            final_sum += wg_sum[i];
        }
        d_coeffs[basis_idx] += final_sum;
    }
}
