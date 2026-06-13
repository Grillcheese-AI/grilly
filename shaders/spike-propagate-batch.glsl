#version 450
// spike-propagate-batch.glsl — event-driven sparse propagation for a BATCH of
// M spike vectors through one (N_in x N_out) weight matrix, single dispatch.
// out[m, post] = sum over fired pre of vector m of W[pre*N_out + post].
// Thread t -> (m = t/N_out, post = t%N_out); loops that vector's fired slice.
// One dispatch over M*N_out covers the whole batch (W stays resident; caller
// records it in a single submit). Non-square: pre in [0,N_in), post in [0,N_out).
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer FiredIdx     { uint  fired_idx[]; };     // concat
layout(set = 0, binding = 1) readonly buffer FiredOffsets { uint  fired_offsets[]; }; // [M]
layout(set = 0, binding = 2) readonly buffer FiredCounts  { uint  fired_counts[]; };  // [M]
layout(set = 0, binding = 3) readonly buffer Weights      { float W[]; };             // [N_in*N_out]
layout(set = 0, binding = 4)          buffer Output       { float O[]; };             // [M*N_out]
layout(set = 0, binding = 5) readonly buffer FiredVals    { float fired_vals[]; };    // concat, spike magnitudes

layout(push_constant) uniform PushConsts { uint N_out; uint M; };

void main() {
    uint t = gl_GlobalInvocationID.x;
    if (t >= M * N_out) return;
    uint m    = t / N_out;
    uint post = t % N_out;
    uint off  = fired_offsets[m];
    uint cnt  = fired_counts[m];
    float acc = 0.0;
    for (uint i = 0u; i < cnt; ++i) {
        uint pre = fired_idx[off + i];
        acc += fired_vals[off + i] * W[pre * N_out + post];
    }
    O[m * N_out + post] = acc;
}
