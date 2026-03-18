#version 450
layout(local_size_x = 256) in;
layout(set=0, binding=0) readonly buffer Query { uint query[]; };
layout(set=0, binding=1) readonly buffer Codebook { uint codebook[]; };  // num_entries * words_per_vec
layout(set=0, binding=2) writeonly buffer Sims { float similarities[]; };
layout(push_constant) uniform Params {
    uint words_per_vec;  // dim / 32
    uint num_entries;    // codebook size
    uint dim;            // original dimension
};

void main() {
    uint entry_idx = gl_GlobalInvocationID.x;
    if (entry_idx >= num_entries) return;

    uint hamming = 0u;
    uint base = entry_idx * words_per_vec;
    for (uint w = 0u; w < words_per_vec; w++) {
        hamming += bitCount(query[w] ^ codebook[base + w]);
    }
    similarities[entry_idx] = 1.0 - float(hamming) / float(dim);
}
