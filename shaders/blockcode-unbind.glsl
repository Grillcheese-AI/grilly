#version 450
layout(local_size_x = 256) in;
layout(set=0, binding=0) readonly buffer Composite { float composite[]; };
layout(set=0, binding=1) readonly buffer Key { float key[]; };
layout(set=0, binding=2) writeonly buffer Out { float out_data[]; };
layout(push_constant) uniform Params {
    uint num_blocks;
    uint block_size;
    uint batch_size;
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint total_blocks = batch_size * num_blocks;
    if (idx >= total_blocks) return;

    uint batch_idx = idx / num_blocks;
    uint block_idx = idx % num_blocks;
    uint base = (batch_idx * num_blocks + block_idx) * block_size;

    uint hot_comp = 0u;
    uint hot_key = 0u;
    for (uint i = 0u; i < block_size; i++) {
        if (composite[base + i] > 0.5) hot_comp = i;
        if (key[base + i] > 0.5) hot_key = i;
    }

    // Inverse (correlation): (hot_comp - hot_key + block_size) % block_size
    uint hot_out = (hot_comp - hot_key + block_size) % block_size;
    for (uint i = 0u; i < block_size; i++) {
        out_data[base + i] = (i == hot_out) ? 1.0 : 0.0;
    }
}
