#version 450

// Binarization & Packing: float32 LSH output → packed uint32 binary VSA.
//
// Each thread owns one uint (32 bits). Reads 32 consecutive floats,
// applies threshold (>0 → 1), packs into a single uint32.
// Entire 10,240-dim vector packed in one dispatch with zero VRAM round-trip.
//
// Dispatch: workgroups_x = ceil(VECTOR_UINTS / 64)

layout(constant_id = 0) const uint VECTOR_DIM = 10240;
layout(constant_id = 1) const uint VECTOR_UINTS = 320;  // ceil(10240 / 32)

layout (local_size_x = 64) in;  // Wave64 optimized

layout(std430, binding = 0) readonly buffer LSHOutput { float continuous_vector[]; };
layout(std430, binding = 1) writeonly buffer VSAPacked { uint packed_binary[]; };

void main() {
    uint uint_idx = gl_GlobalInvocationID.x;
    if (uint_idx >= VECTOR_UINTS) return;

    uint packed_val = 0u;
    uint base_float_idx = uint_idx * 32u;

    for (uint i = 0u; i < 32u; i++) {
        uint float_idx = base_float_idx + i;
        if (float_idx < VECTOR_DIM) {
            if (continuous_vector[float_idx] > 0.0) {
                packed_val |= (1u << i);
            }
        }
    }

    packed_binary[uint_idx] = packed_val;
}
