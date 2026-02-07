#version 450

// Resonator step: project an unbound vector onto a codebook
// Outputs similarity scores for each codebook entry.

layout (local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer Query {
    float query[];
};

layout(set = 0, binding = 1) readonly buffer Codebook {
    float codebook[];  // (codebook_size * dim)
};

layout(set = 0, binding = 2) buffer Similarities {
    float similarities[];
};

layout(push_constant) uniform PushConsts {
    uint dim;
    uint codebook_size;
};

shared float partial_dot[64];

void main() {
    uint vec_idx = gl_WorkGroupID.x;
    uint local_idx = gl_LocalInvocationID.x;

    if (vec_idx >= codebook_size) {
        return;
    }

    float dot_sum = 0.0;
    for (uint i = local_idx; i < dim; i += 64) {
        float q_val = query[i];
        float v_val = codebook[vec_idx * dim + i];
        dot_sum += q_val * v_val;
    }

    partial_dot[local_idx] = dot_sum;
    barrier();

    for (uint stride = 32; stride > 0; stride >>= 1) {
        if (local_idx < stride) {
            partial_dot[local_idx] += partial_dot[local_idx + stride];
        }
        barrier();
    }

    if (local_idx == 0) {
        if (dim > 0) {
            similarities[vec_idx] = partial_dot[0] / float(dim);
        } else {
            similarities[vec_idx] = 0.0;
        }
    }
}
