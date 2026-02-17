#version 450

// Fused selective-scan math for SSM blocks.
// One invocation processes a (batch, feature) lane across the full sequence.
// This keeps the recurrence on GPU and removes CPU-side np.cumsum bottlenecks.

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer GateBuffer {
    float gate[];
};

layout(set = 0, binding = 1) readonly buffer ValueBuffer {
    float value[];
};

layout(set = 0, binding = 2) readonly buffer DecayBuffer {
    float decay[];
};

layout(set = 0, binding = 3) writeonly buffer OutputBuffer {
    float scan_out[];
};

layout(push_constant) uniform PushConstants {
    uint batch_size;
    uint seq_len;
    uint features;
    float min_decay;
    float max_decay;
} pc;

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

    for (uint t = 0; t < pc.seq_len; ++t) {
        uint idx = ((b * pc.seq_len + t) * pc.features) + f;
        float g = gate[idx];
        float v = value[idx];
        state = (d * state) + (keep * v);
        scan_out[idx] = state * g;
    }
}
