#version 450
layout(local_size_x = 256) in;
layout(set=0, binding=0) readonly buffer A { float a[]; };
layout(set=0, binding=1) readonly buffer B { float b[]; };
layout(set=0, binding=2) writeonly buffer Out { float out_data[]; };
layout(push_constant) uniform Params {
    uint num_blocks;   // k
    uint block_size;   // l
    uint batch_size;   // number of vectors in batch
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint total_blocks = batch_size * num_blocks;
    if (idx >= total_blocks) return;

    uint batch_idx = idx / num_blocks;
    uint block_idx = idx % num_blocks;
    uint base = (batch_idx * num_blocks + block_idx) * block_size;

    // Find hot positions in a and b for this block
    uint hot_a = 0u;
    uint hot_b = 0u;
    for (uint i = 0u; i < block_size; i++) {
        if (a[base + i] > 0.5) hot_a = i;
        if (b[base + i] > 0.5) hot_b = i;
    }

    // Result: one-hot at (hot_a + hot_b) % block_size
    uint hot_out = (hot_a + hot_b) % block_size;
    for (uint i = 0u; i < block_size; i++) {
        out_data[base + i] = (i == hot_out) ? 1.0 : 0.0;
    }
}
