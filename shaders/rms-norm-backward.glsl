#version 450

// RMSNorm backward (2-pass, no atomics — each thread owns a unique output cell).
// Forward:  y_i = w_f * x_i * r,   r = inversesqrt(mean(x^2) + eps),
//           mean(x^2) = (1/n) * sum_j x_j^2   (n = features, per row)
//
// Backward (incoming g_i = dL/dy_i), per row of `features` elements:
//   c = sum_j (g_j * w_{f(j)} * x_j)
//   dL/dx_i = r * w_{f(i)} * g_i  -  (r^3 / n) * x_i * c
//   dL/dw_f = sum_positions (g * x * r)
//
// pass_type == 0: grad_input  (one thread per element, batch*seq*features)
// pass_type == 1: grad_weight (one thread per feature, features)

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer GradOutput {
    float grad_output[];
};
layout(set = 0, binding = 1) readonly buffer Input {
    float input_data[];
};
layout(set = 0, binding = 2) readonly buffer Weight {
    float weight[];
};
layout(set = 0, binding = 3) buffer GradInput {
    float grad_input[];
};
layout(set = 0, binding = 4) buffer GradWeight {
    float grad_weight[];
};

layout(push_constant) uniform PushConsts {
    uint batch_size;
    uint seq_len;
    uint features;
    float eps;
    uint pass_type;
};

void main() {
    uint gID = gl_GlobalInvocationID.x;
    uint n = features;
    uint total_positions = batch_size * seq_len;

    if (pass_type == 0) {
        // grad_input: one thread per element.
        uint total_elements = total_positions * features;
        if (gID >= total_elements) return;

        uint pos_idx = gID / features;
        uint feat_idx = gID % features;
        uint row_base = pos_idx * features;

        float sum_sq = 0.0;
        float c = 0.0;
        for (uint j = 0; j < n; j++) {
            float xj = input_data[row_base + j];
            sum_sq += xj * xj;
            c += grad_output[row_base + j] * weight[j] * xj;
        }
        float r = inversesqrt(sum_sq / float(n) + eps);

        float x_i = input_data[gID];
        float g_i = grad_output[gID];
        float w_i = weight[feat_idx];
        float r3 = r * r * r;

        grad_input[gID] = r * w_i * g_i - (r3 / float(n)) * x_i * c;

    } else if (pass_type == 1) {
        // grad_weight: one thread per feature, sum over all positions.
        if (gID >= features) return;
        uint feat_idx = gID;

        float gw = 0.0;
        for (uint p = 0; p < total_positions; p++) {
            uint row_base = p * features;
            // recompute r for row p
            float sum_sq = 0.0;
            for (uint j = 0; j < n; j++) {
                float xj = input_data[row_base + j];
                sum_sq += xj * xj;
            }
            float r = inversesqrt(sum_sq / float(n) + eps);
            uint idx = row_base + feat_idx;
            gw += grad_output[idx] * input_data[idx] * r;
        }
        grad_weight[feat_idx] = gw;
    }
}
