#version 450

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Input tensor
layout(set = 0, binding = 0) readonly buffer Input {
    float input_data[];
};

// Output tensor
layout(set = 0, binding = 1) buffer Output {
    float output_data[];
};

// Scale weights (features)
layout(set = 0, binding = 2) readonly buffer Weight {
    float weight[];
};

// RMS values buffer (batch * seq)
layout(set = 0, binding = 3) buffer RMSBuffer {
    float rms_vals[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint batch_size;
    uint seq_len;
    uint features;
    float eps;
    uint pass_type;       // 0 = compute RMS, 1 = normalize
};

void main() {
    uint gID = gl_GlobalInvocationID.x;

    if (pass_type == 0) {
        // Pass 0: Compute RMS = sqrt(mean(x^2))
        // Store mean(x^2) for now, take sqrt during normalize
        uint total_positions = batch_size * seq_len;
        if (gID >= total_positions) return;

        uint batch_idx = gID / seq_len;
        uint seq_idx = gID % seq_len;

        float sum_sq = 0.0;
        for (uint f = 0; f < features; f++) {
            uint idx = batch_idx * seq_len * features + seq_idx * features + f;
            float x = input_data[idx];
            sum_sq += x * x;
        }
        rms_vals[gID] = sum_sq / float(features);

    } else if (pass_type == 1) {
        // Pass 1: Normalize: out = weight * x * rsqrt(mean_sq + eps)
        uint total_elements = batch_size * seq_len * features;
        if (gID >= total_elements) return;

        uint batch_idx = gID / (seq_len * features);
        uint remainder = gID % (seq_len * features);
        uint seq_idx = remainder / features;
        uint feat_idx = remainder % features;

        uint pos_idx = batch_idx * seq_len + seq_idx;
        float mean_sq = rms_vals[pos_idx];
        float inv_rms = inversesqrt(mean_sq + eps);

        float x = input_data[gID];
        output_data[gID] = weight[feat_idx] * x * inv_rms;
    }
}
