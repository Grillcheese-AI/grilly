#version 450

// Multiplication-free softmax: relu(x - max) / sum(relu(x - max)) — no exp().
// Same 3-pass layout as activation-softmax.glsl; pass 2 sums positive parts only.

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer Input {
    float input_data[];
};

layout(set = 0, binding = 1) buffer Output {
    float output_data[];
};

layout(set = 0, binding = 2) buffer MaxValues {
    float max_vals[];
};

layout(set = 0, binding = 3) buffer SumPos {
    float sum_pos[];
};

layout(push_constant) uniform PushConsts {
    uint batch_size;
    uint seq_len;
    uint features;
    uint pass_type;
    uint dim;
};

void main() {
    uint gID = gl_GlobalInvocationID.x;

    if (pass_type == 0u) {
        uint total_positions = batch_size * seq_len;
        if (gID >= total_positions) return;

        uint batch_idx = gID / seq_len;
        uint seq_idx = gID % seq_len;

        float max_val = -1e10;
        for (uint f = 0; f < features; f++) {
            uint idx = batch_idx * seq_len * features + seq_idx * features + f;
            max_val = max(max_val, input_data[idx]);
        }
        max_vals[gID] = max_val;

    } else if (pass_type == 1u) {
        uint total_positions = batch_size * seq_len;
        if (gID >= total_positions) return;

        uint batch_idx = gID / seq_len;
        uint seq_idx = gID % seq_len;
        float max_val = max_vals[gID];

        float sum = 0.0;
        for (uint f = 0; f < features; f++) {
            uint idx = batch_idx * seq_len * features + seq_idx * features + f;
            float v = input_data[idx] - max_val;
            sum += max(v, 0.0);
        }
        sum_pos[gID] = sum;

    } else if (pass_type == 2u) {
        uint total_elements = batch_size * seq_len * features;
        if (gID >= total_elements) return;

        uint batch_idx = gID / (seq_len * features);
        uint remainder = gID % (seq_len * features);
        uint seq_idx = remainder / features;

        uint pos_idx = batch_idx * seq_len + seq_idx;
        float max_val = max_vals[pos_idx];
        float sum = sum_pos[pos_idx];

        float val = input_data[gID];
        float z = max(val - max_val, 0.0);
        if (sum < 1e-5) {
            output_data[gID] = 1.0 / float(features);
        } else {
            float denom = max(sum, 1e-6);
            output_data[gID] = z / denom;
        }
    }
}
