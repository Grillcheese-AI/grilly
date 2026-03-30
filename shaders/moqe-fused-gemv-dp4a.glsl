#version 450
#extension GL_KHR_shader_subgroup_arithmetic : enable
#extension GL_EXT_shader_atomic_float : enable
// GL_EXT_shader_integer_dot_product — use manual dot4 for portability

// Ultimate MoQE Fused GEMV with DP4a (4x throughput).
//
// Each thread processes 4 activations via hardware v_dot4_i32_i8.
// Wave64 = 256 dimensions per cycle. Combined with MoQE routing:
// - Expert 0 (4-bit): sign-extend INT4 → INT8, feed to same DP4a
// - Expert 1 (8-bit): unpack INT8 directly
//
// Zero warp divergence for autoregressive generation (batch=1).
//
// Dispatch: X = num_blocks_per_row, Y = num_output_rows

layout(constant_id = 0) const uint WAVE_SIZE = 64;
layout(local_size_x_id = 0) in;

layout(std430, binding = 0) readonly buffer FP32Activations { float activations[]; };
layout(std430, binding = 1) readonly buffer RouterChoice    { uint choices[]; };
layout(std430, binding = 2) readonly buffer Exp0_INT4       { uint e0_weights[]; };
layout(std430, binding = 3) readonly buffer Exp1_INT8       { uint e1_weights[]; };
layout(std430, binding = 4) readonly buffer ScalesE0        { float w_scales_e0[]; };
layout(std430, binding = 5) readonly buffer ScalesE1        { float w_scales_e1[]; };
layout(std430, binding = 6) buffer FP32Output               { float output_vector[]; };

ivec4 unpack_int8_to_ivec4(uint p) {
    return ivec4(
        (int(p) << 24) >> 24,
        (int(p) << 16) >> 24,
        (int(p) << 8)  >> 24,
        int(p) >> 24
    );
}

ivec4 unpack_int4_to_ivec4(uint p, uint half_idx) {
    uint chunk = (half_idx == 0) ? (p & 0xFFFFu) : (p >> 16);
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
    uint token_idx  = 0;  // Batch=1 autoregressive

    uint my_expert = choices[token_idx];

    // Read 4 activations per thread (256 dims per wave)
    uint base_idx = (block_idx * WAVE_SIZE + thread_idx) * 4;
    vec4 vals = vec4(
        activations[base_idx],
        activations[base_idx + 1],
        activations[base_idx + 2],
        activations[base_idx + 3]
    );

    // Hardware absmax across 256 values
    vec4 abs_vals = abs(vals);
    float local_max = max(max(abs_vals.x, abs_vals.y), max(abs_vals.z, abs_vals.w));
    float wave_max = max(subgroupMax(local_max), 1e-7);
    float a_scale = wave_max / 127.0;

    ivec4 q_activations = clamp(ivec4(round(vals / a_scale)), ivec4(-127), ivec4(127));

    // Expert-routed weight fetch
    ivec4 q_weights;
    float w_scale;

    if (my_expert == 0) {
        // 4-bit expert: 8 weights per uint, sign-extend to INT8 for DP4a
        uint e0_uint_idx = (row_idx * (gl_NumWorkGroups.x * WAVE_SIZE / 2)) +
                           (block_idx * WAVE_SIZE / 2) + (thread_idx / 2);
        uint packed_w = e0_weights[e0_uint_idx];
        uint half_idx = thread_idx % 2;
        q_weights = unpack_int4_to_ivec4(packed_w, half_idx);
        w_scale = w_scales_e0[row_idx * gl_NumWorkGroups.x + block_idx];
    } else {
        // 8-bit expert: 4 weights per uint, direct unpack
        uint e1_uint_idx = (row_idx * (gl_NumWorkGroups.x * WAVE_SIZE)) +
                           (block_idx * WAVE_SIZE) + thread_idx;
        uint packed_w = e1_weights[e1_uint_idx];
        q_weights = unpack_int8_to_ivec4(packed_w);
        w_scale = w_scales_e1[row_idx * gl_NumWorkGroups.x + block_idx];
    }

    // DP4a: v_dot4_i32_i8 on RDNA 2 — 4 MADs in one instruction
    // Manual DP4a: a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w
    // On RDNA 2 this compiles to v_dot4_i32_i8 when the driver recognizes the pattern
    int thread_dot = q_activations.x * q_weights.x + q_activations.y * q_weights.y
                   + q_activations.z * q_weights.z + q_activations.w * q_weights.w;

    int wave_sum = subgroupAdd(thread_dot);

    if (subgroupElect()) {
        float final_partial_sum = float(wave_sum) * (a_scale * w_scale);
        atomicAdd(output_vector[row_idx], final_partial_sum);
    }
}
