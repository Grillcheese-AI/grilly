#version 450
#extension GL_KHR_shader_subgroup_arithmetic : require

// Fused LayerNorm + Linear projection.
//
// normalize in registers -> project directly -> write output.
// Saves one full VRAM round-trip per transformer layer.
//
// Dispatch: workgroups_x = seq_len

layout(constant_id = 0) const uint D_IN = 768;
layout(constant_id = 1) const uint D_OUT = 768;

layout(local_size_x = 256) in;

layout(std430, set = 0, binding = 0) readonly buffer Input     { float input_data[]; };
layout(std430, set = 0, binding = 1) readonly buffer LNWeight  { float ln_weight[]; };
layout(std430, set = 0, binding = 2) readonly buffer LNBias    { float ln_bias[]; };
layout(std430, set = 0, binding = 3) readonly buffer ProjW     { float proj_w[]; };
layout(std430, set = 0, binding = 4) readonly buffer ProjB     { float proj_b[]; };
layout(std430, set = 0, binding = 5) writeonly buffer Output   { float output_data[]; };

shared float shared_normed[768];
shared float shared_sum;
shared float shared_sq_sum;

void main() {
    uint token_idx = gl_WorkGroupID.x;
    uint tid = gl_LocalInvocationID.x;
    uint input_offset = token_idx * D_IN;
    uint output_offset = token_idx * D_OUT;

    // Init shared atomics
    if (tid == 0u) { shared_sum = 0.0; shared_sq_sum = 0.0; }
    barrier();

    // Mean
    float local_sum = 0.0;
    for (uint i = tid; i < D_IN; i += gl_WorkGroupSize.x) {
        local_sum += input_data[input_offset + i];
    }
    float sg_sum = subgroupAdd(local_sum);
    if (subgroupElect()) { atomicAdd(shared_sum, sg_sum); }
    barrier();
    float mean = shared_sum / float(D_IN);

    // Variance
    float local_sq = 0.0;
    for (uint i = tid; i < D_IN; i += gl_WorkGroupSize.x) {
        float diff = input_data[input_offset + i] - mean;
        local_sq += diff * diff;
    }
    float sg_sq = subgroupAdd(local_sq);
    if (subgroupElect()) { atomicAdd(shared_sq_sum, sg_sq); }
    barrier();
    float inv_std = inversesqrt(shared_sq_sum / float(D_IN) + 1e-6);

    // Normalize + affine -> shared memory (no VRAM write)
    for (uint i = tid; i < D_IN; i += gl_WorkGroupSize.x) {
        float normed = (input_data[input_offset + i] - mean) * inv_std;
        shared_normed[i] = normed * ln_weight[i] + ln_bias[i];
    }
    barrier();

    // Linear projection from shared memory
    for (uint o = tid; o < D_OUT; o += gl_WorkGroupSize.x) {
        float acc = proj_b[o];
        for (uint i = 0u; i < D_IN; i++) {
            acc += proj_w[o * D_IN + i] * shared_normed[i];
        }
        output_data[output_offset + o] = acc;
    }
}
