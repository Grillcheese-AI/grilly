#version 450
layout(local_size_x = 256) in;
layout(set=0, binding=0) readonly buffer A { uint a[]; };
layout(set=0, binding=1) readonly buffer B { uint b[]; };
layout(set=0, binding=2) writeonly buffer Out { uint out_data[]; };
layout(push_constant) uniform Params { uint num_words; };  // total uint32 words (dim/32 * batch)

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= num_words) return;
    out_data[idx] = a[idx] ^ b[idx];
}
