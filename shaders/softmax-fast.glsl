#version 450
#extension GL_KHR_shader_subgroup_arithmetic : enable
#extension GL_KHR_shader_subgroup_shuffle : enable

// RDNA2-optimized softmax: subgroup reductions + online max rescaling.
// 256 threads per workgroup (8 subgroups of 32 on Wave32).
// Numerically stable — max subtraction built into the reduction.
//
// Each workgroup processes one row of length <= 256.
// For longer rows, dispatch multiple workgroups per row (requires cross-WG reduction).

layout(local_size_x = 256) in;

layout(set = 0, binding = 0) buffer Input  { float data[]; } In;
layout(set = 0, binding = 1) buffer Output { float data[]; } Out;

layout(push_constant) uniform PushConsts {
    uint batch_size;    // number of rows
    uint row_length;    // elements per row
};

shared float lds_max[8];
shared float lds_sum[8];

void main() {
    uint row = gl_WorkGroupID.x;
    uint l_idx = gl_LocalInvocationID.x;
    uint sg_id = gl_SubgroupInvocationID;
    uint sg_idx = gl_SubgroupID;

    if (row >= batch_size) return;

    uint base = row * row_length;

    // Load value (pad with -inf for threads beyond row_length)
    float val = (l_idx < row_length) ? In.data[base + l_idx] : -1e38;

    // --- STEP 1: Subgroup-Level Online Reduction ---
    float m = subgroupMax(val);
    float d = subgroupAdd(exp(val - m));

    // --- STEP 2: Workgroup-Level Reduction via LDS ---
    if (subgroupElect()) {
        lds_max[sg_idx] = m;
        lds_sum[sg_idx] = d;
    }
    barrier();

    // Final reduction by first subgroup
    if (sg_idx == 0) {
        float final_m = -1e38;
        float final_d = 0.0;

        uint n_sgs = (row_length + 31) / 32;
        if (n_sgs > 8) n_sgs = 8;

        for (uint i = 0; i < n_sgs; ++i) {
            float prev_m = final_m;
            final_m = max(final_m, lds_max[i]);
            final_d = final_d * exp(prev_m - final_m) + lds_sum[i] * exp(lds_max[i] - final_m);
        }

        if (sg_id == 0) {
            lds_max[0] = final_m;
            lds_sum[0] = final_d;
        }
    }
    barrier();

    // --- STEP 3: Final Computation ---
    float wg_max = lds_max[0];
    float wg_sum = lds_sum[0];

    if (l_idx < row_length) {
        Out.data[base + l_idx] = exp(val - wg_max) / wg_sum;
    }
}
