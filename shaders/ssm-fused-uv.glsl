#version 450

// Fused SSM selective-scan from projected UV tensor.
// Input uv layout: [batch, seq_len, 2 * features]
// - first half  => gate pre-activation
// - second half => value pre-activation
//
// Output layout: [batch, seq_len, features]
// state_t = d * state_{t-1} + (1-d) * tanh(value_t)
// out_t   = state_t * sigmoid(gate_t)

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer UVBuffer {
    float uv[];
};

layout(set = 0, binding = 1) readonly buffer DecayBuffer {
    float decay[];
};

layout(set = 0, binding = 2) readonly buffer MaskBuffer {
    float mask[];
};

layout(set = 0, binding = 3) writeonly buffer OutputBuffer {
    float scan_out[];
};

layout(push_constant) uniform PushConstants {
    uint batch_size;
    uint seq_len;
    uint features;
    uint has_mask;
    float min_decay;
    float max_decay;
} pc;

float safe_sigmoid(float x) {
    float xc = clamp(x, -20.0, 20.0);
    return 1.0 / (1.0 + exp(-xc));
}

float safe_tanh(float x) {
    return tanh(clamp(x, -20.0, 20.0));
}

void main() {
    uint lane = gl_GlobalInvocationID.x;
    uint total_lanes = pc.batch_size * pc.features;
    if (lane >= total_lanes) {
        return;
    }

    uint b = lane / pc.features;
    uint f = lane % pc.features;

    float d = clamp(decay[f], pc.min_decay, pc.max_decay);
    float keep = 1.0 - d;
    float state = 0.0;
    uint uv_stride = 2u * pc.features;

    for (uint t = 0; t < pc.seq_len; ++t) {
        uint base_uv = (b * pc.seq_len + t) * uv_stride;
        float g = safe_sigmoid(uv[base_uv + f]);
        float v = safe_tanh(uv[base_uv + pc.features + f]);

        if (pc.has_mask != 0u) {
            float m = mask[b * pc.seq_len + t];
            g *= m;
            v *= m;
        }

        state = (d * state) + (keep * v);
        uint out_idx = (b * pc.seq_len + t) * pc.features + f;
        scan_out[out_idx] = state * g;
    }
}
