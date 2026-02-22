#version 450

// Fused RMSNorm + Linear: output = Linear(RMSNorm(x))
// Eliminates intermediate buffer between normalization and projection

layout (local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer Input {
    float input_data[];
};

layout(set = 0, binding = 1) readonly buffer NormWeight {
    float norm_weight[];
};

layout(set = 0, binding = 2) readonly buffer LinearWeight {
    float W[];
};

layout(set = 0, binding = 3) buffer Output {
    float output_data[];
};

layout(set = 0, binding = 4) buffer RMSBuffer {
    float rms_vals[];
};

layout(push_constant) uniform PushConsts {
    uint batch_seq;
    uint input_dim;
    uint output_dim;
    float eps;
    uint pass_type;    // 0 = compute RMS, 1 = fused normalize + linear
};

void main() {
    if (pass_type == 0) {
        // Pass 0: Compute mean(x^2) for each position
        uint pos = gl_GlobalInvocationID.x;
        if (pos >= batch_seq) return;

        float sum_sq = 0.0;
        for (uint f = 0; f < input_dim; f++) {
            float x = input_data[pos * input_dim + f];
            sum_sq += x * x;
        }
        rms_vals[pos] = sum_sq / float(input_dim);

    } else if (pass_type == 1) {
        // Pass 1: Fused RMSNorm + Linear
        uint row = gl_GlobalInvocationID.y;  // batch_seq index
        uint col = gl_GlobalInvocationID.x;  // output_dim index

        if (row >= batch_seq || col >= output_dim) return;

        float inv_rms = inversesqrt(rms_vals[row] + eps);

        float sum = 0.0;
        for (uint k = 0; k < input_dim; k++) {
            float x = input_data[row * input_dim + k];
            float normed = norm_weight[k] * x * inv_rms;
            sum += W[col * input_dim + k] * normed;
        }

        output_data[row * output_dim + col] = sum;
    }
}
