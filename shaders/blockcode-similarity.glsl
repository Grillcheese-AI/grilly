#version 450
layout(local_size_x = 256) in;
layout(set=0, binding=0) readonly buffer Query { float query[]; };
layout(set=0, binding=1) readonly buffer Codebook { float codebook[]; };
layout(set=0, binding=2) writeonly buffer Sims { float similarities[]; };
layout(push_constant) uniform Params {
    uint num_blocks;    // k
    uint block_size;    // l
    uint num_entries;   // codebook size
};

void main() {
    uint entry_idx = gl_GlobalInvocationID.x;
    if (entry_idx >= num_entries) return;

    float sim = 0.0;
    uint entry_base = entry_idx * num_blocks * block_size;
    for (uint b = 0u; b < num_blocks; b++) {
        uint q_base = b * block_size;
        uint e_base = entry_base + b * block_size;
        // For one-hot blocks, dot product is 1 if same position, 0 otherwise
        for (uint i = 0u; i < block_size; i++) {
            sim += query[q_base + i] * codebook[e_base + i];
        }
    }
    similarities[entry_idx] = sim / float(num_blocks);
}
