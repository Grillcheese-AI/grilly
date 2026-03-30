#version 450
#extension GL_KHR_shader_subgroup_arithmetic : enable
#extension GL_EXT_shader_atomic_float : enable

// MoQE Fused GEMV with DP4a pattern — optimized for RDNA 2.
//
// Fixes applied:
//   1. vec4 activation loads (128-bit, single fetch per thread)
//   2. Inverse scale multiplication (no float division)
//   3. Branchless INT4 unpacking (pure ALU, no ternary)
//   4. Router choice via push constant (SGPR, not SSBO)
//   5. Explicit subgroup_size comment (enforce Wave64 via host)
//
// DP4a pattern: manual i32 dot product that compilers recognize as v_dot4_i32_i8.
// For guaranteed DP4a, add GL_EXT_shader_explicit_arithmetic_types_int8 when
// the Vulkan SDK supports it on RDNA 2.

layout(constant_id = 0) const uint WAVE_SIZE = 64;
layout(local_size_x_id = 0) in;

// vec4 activation buffer for 128-bit coalesced loads
layout(std430, binding = 0) readonly buffer FP32Activations { vec4 activations[]; };
layout(std430, binding = 1) readonly buffer Exp0_INT4       { uint e0_weights[]; };
layout(std430, binding = 2) readonly buffer Exp1_INT8       { uint e1_weights[]; };
layout(std430, binding = 3) readonly buffer ScalesE0        { float w_scales_e0[]; };
layout(std430, binding = 4) readonly buffer ScalesE1        { float w_scales_e1[]; };
layout(std430, binding = 5) buffer FP32Output               { float output_vector[]; };

layout(push_constant) uniform PushConsts {
    uint my_expert;   // Router choice: 0=4-bit, 1=8-bit (passed directly, not from SSBO)
};

ivec4 unpack_int8_to_ivec4(uint p) {
    return ivec4(
        (int(p) << 24) >> 24,
        (int(p) << 16) >> 24,
        (int(p) << 8)  >> 24,
        int(p) >> 24
    );
}

ivec4 unpack_int4_to_ivec4(uint p, uint half_idx) {
    // Branchless: half_idx * 16 gives shift of 0 or 16
    uint chunk = (p >> (half_idx * 16u)) & 0xFFFFu;
    return ivec4(
        (int(chunk) << 28) >> 28,
        (int(chunk) << 24) >> 28,
        (int(chunk) << 20) >> 28,
        (int(chunk) << 16) >> 28
    );
}

void main() {
    uint thread_idx = gl_LocalInvocationID.x;
    uint row_idx    = gl_WorkGroupID.y;
    uint block_idx  = gl_WorkGroupID.x;

    // 1. vec4 activation load: single 128-bit fetch per thread
    uint vec4_idx = block_idx * WAVE_SIZE + thread_idx;
    vec4 vals = activations[vec4_idx];

    // 2. Hardware absmax via subgroup reduction
    vec4 abs_vals = abs(vals);
    float local_max = max(max(abs_vals.x, abs_vals.y), max(abs_vals.z, abs_vals.w));
    float wave_max = max(subgroupMax(local_max), 1e-7);

    // 3. Inverse scale: multiply instead of divide (saves ~4 ALU cycles)
    float a_inv_scale = 127.0 / wave_max;
    float a_scale = wave_max / 127.0;

    ivec4 q_activations = clamp(ivec4(round(vals * a_inv_scale)), ivec4(-127), ivec4(127));

    // 4. Expert-routed weight fetch (router choice is push constant → SGPR)
    ivec4 q_weights;
    float w_scale;

    if (my_expert == 0u) {
        // 4-bit expert: branchless half_idx selection
        uint e0_uint_idx = (row_idx * (gl_NumWorkGroups.x * WAVE_SIZE / 2)) +
                           (block_idx * WAVE_SIZE / 2) + (thread_idx / 2);
        uint packed_w = e0_weights[e0_uint_idx];
        uint half_idx = thread_idx & 1u;  // 0 or 1

        q_weights = unpack_int4_to_ivec4(packed_w, half_idx);
        w_scale = w_scales_e0[row_idx * gl_NumWorkGroups.x + block_idx];
    } else {
        // 8-bit expert
        uint e1_uint_idx = (row_idx * (gl_NumWorkGroups.x * WAVE_SIZE)) +
                           (block_idx * WAVE_SIZE) + thread_idx;
        uint packed_w = e1_weights[e1_uint_idx];
        q_weights = unpack_int8_to_ivec4(packed_w);
        w_scale = w_scales_e1[row_idx * gl_NumWorkGroups.x + block_idx];
    }

    // 5. DP4a: 4 MADs in one pattern (compiler maps to v_dot4_i32_i8 on RDNA 2)
    int thread_dot = q_activations.x * q_weights.x + q_activations.y * q_weights.y
                   + q_activations.z * q_weights.z + q_activations.w * q_weights.w;

    // 6. Subgroup reduction + output
    int wave_sum = subgroupAdd(thread_dot);

    if (subgroupElect()) {
        float final_partial_sum = float(wave_sum) * (a_scale * w_scale);
        atomicAdd(output_vector[row_idx], final_partial_sum);
    }
}
