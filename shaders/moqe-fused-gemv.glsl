#version 450
#extension GL_KHR_shader_subgroup_arithmetic : enable
#extension GL_EXT_shader_atomic_float : enable

// Fused Dynamic Quantization + GEMV.
//
// Activations never leave GPU registers: quantized on-the-fly, multiplied
// against pre-quantized INT8 weights, accumulated via subgroupAdd.
// Zero VRAM round-trip for activation quantization.
//
// Wave-size adaptive via specialization constant.
//
// Dispatch: X = num_blocks_per_row, Y = num_output_rows

layout(constant_id = 0) const uint WAVE_SIZE = 64;
layout(local_size_x_id = 0) in;

layout(std430, binding = 0) readonly buffer FP32Activations { float activations[]; };
layout(std430, binding = 1) readonly buffer INT8Weights     { uint packed_weights[]; };
layout(std430, binding = 2) readonly buffer WeightScales    { float w_scales[]; };
layout(std430, binding = 3) buffer FP32Output               { float output_vector[]; };

int unpack_int8(uint packed_data, uint byte_index) {
    uint shift = byte_index * 8;
    uint masked = (packed_data >> shift) & 0xFFu;
    return int(masked) - ((int(masked) & 128) << 1);
}

void main() {
    uint thread_idx = gl_LocalInvocationID.x;
    uint row_idx    = gl_WorkGroupID.y;
    uint block_idx  = gl_WorkGroupID.x;

    // Dynamic quantization in registers
    uint activation_idx = block_idx * WAVE_SIZE + thread_idx;
    float val = activations[activation_idx];

    float abs_val = abs(val);
    float max_val = max(subgroupMax(abs_val), 1e-7);
    float a_scale = max_val / 127.0;

    int q_activation = clamp(int(round(val / a_scale)), -127, 127);

    // Fetch pre-quantized weights
    uint uints_per_block = WAVE_SIZE / 4;
    uint weight_uint_idx = (row_idx * (gl_NumWorkGroups.x * uints_per_block)) +
                           (block_idx * uints_per_block) +
                           (thread_idx / 4);

    uint packed_w = packed_weights[weight_uint_idx];
    uint byte_offset = thread_idx % 4;
    int q_weight = unpack_int8(packed_w, byte_offset);

    // Integer dot product + subgroup reduction
    int local_dot = q_activation * q_weight;
    int subgroup_sum = subgroupAdd(local_dot);

    if (subgroupElect()) {
        float w_scale = w_scales[row_idx * gl_NumWorkGroups.x + block_idx];
        float final_partial_sum = float(subgroup_sum) * (a_scale * w_scale);
        atomicAdd(output_vector[row_idx], final_partial_sum);
    }
}
