#version 450
#extension GL_KHR_shader_subgroup_arithmetic : enable

// Dynamic block-wise quantization: FP32 → INT8 on-the-fly.
//
// Each subgroup = one quantization block.
// subgroupMax finds absmax in a single cycle (no shared memory barriers).
// Scale = absmax / 127. Quantized values packed 4-per-uint.
//
// Dispatch: workgroups_x = num_activation_blocks

#define BLOCK_SIZE 32
layout (local_size_x = BLOCK_SIZE) in;

layout(std430, binding = 0) readonly buffer FP32Activations { float activations[]; };
layout(std430, binding = 1) writeonly buffer INT8Packed     { uint quantized_data[]; };
layout(std430, binding = 2) writeonly buffer Scales         { float block_scales[]; };

void main() {
    uint global_idx = gl_GlobalInvocationID.x;
    uint block_idx = gl_WorkGroupID.x;
    uint local_idx = gl_LocalInvocationID.x;

    float val = activations[global_idx];

    // Hardware absmax across subgroup (single cycle on RDNA 2)
    float abs_val = abs(val);
    float max_val = subgroupMax(abs_val);
    max_val = max(max_val, 1e-7);

    float scale = 127.0 / max_val;

    int q_val = int(round(val * scale));
    q_val = clamp(q_val, -127, 127);

    // Pack 4 INT8s into one uint via atomicOr
    uint byte_shift = (local_idx % 4) * 8;
    uint masked_byte = uint(q_val) & 0xFFu;
    atomicOr(quantized_data[global_idx / 4], masked_byte << byte_shift);

    if (subgroupElect()) {
        block_scales[block_idx] = max_val / 127.0;
    }
}
