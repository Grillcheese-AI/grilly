#version 450

// VSA Block-Code Argmax — discretize continuous logits to one-hot.
//
// For each spatial position and each of k blocks, finds the max
// value across l elements and sets it to 1.0, all others to 0.0.
// Replaces Gumbel-Softmax at inference time.
//
// Dispatch: (num_positions, 1, 1) with local_size_z = k

layout(local_size_x = 1, local_size_y = 1, local_size_z = 8) in;

layout(std430, set = 0, binding = 0) readonly buffer InputBuffer {
    float in_data[];
};

layout(std430, set = 0, binding = 1) writeonly buffer OutputBuffer {
    float out_data[];
};

layout(push_constant) uniform PushConstants {
    uint num_positions;
    uint k;
    uint l;
};

void main() {
    uint pos = gl_WorkGroupID.x;
    uint block_k = gl_LocalInvocationID.z;

    if (pos >= num_positions || block_k >= k) return;

    uint base_idx = (pos * k * l) + (block_k * l);

    // Find argmax in this block
    float max_val = -1e30;
    uint max_idx = 0;

    for (uint i = 0; i < l; i++) {
        float val = in_data[base_idx + i];
        if (val > max_val) {
            max_val = val;
            max_idx = i;
        }
    }

    // Write one-hot
    for (uint i = 0; i < l; i++) {
        out_data[base_idx + i] = (i == max_idx) ? 1.0 : 0.0;
    }
}
