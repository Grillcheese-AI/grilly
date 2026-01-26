#version 450

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Gradient w.r.t. output (from upstream)
layout(set = 0, binding = 0) readonly buffer GradOutput {
    float grad_output[];
};

// Original input (cached from forward)
layout(set = 0, binding = 1) readonly buffer Input {
    float input_data[];
};

// Gamma (scale) parameters
layout(set = 0, binding = 2) readonly buffer Gamma {
    float gamma[];
};

// Mean values (from forward pass)
layout(set = 0, binding = 3) readonly buffer MeanBuffer {
    float mean_vals[];
};

// Variance values (from forward pass)
layout(set = 0, binding = 4) readonly buffer VarianceBuffer {
    float var_vals[];
};

// Output: gradient w.r.t. input
layout(set = 0, binding = 5) buffer GradInput {
    float grad_input[];
};

// Output: gradient w.r.t. gamma (atomic add across batch)
layout(set = 0, binding = 6) buffer GradGamma {
    float grad_gamma[];
};

// Output: gradient w.r.t. beta (atomic add across batch)
layout(set = 0, binding = 7) buffer GradBeta {
    float grad_beta[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint batch_size;
    uint seq_len;
    uint features;
    float eps;
    uint pass_type;  // 0 = compute stats, 1 = compute grad_input, 2 = compute grad_gamma/beta
};

// Shared memory for reductions
shared float shared_sum[256];
shared float shared_sum2[256];

void main() {
    uint gID = gl_GlobalInvocationID.x;
    uint lID = gl_LocalInvocationID.x;

    if (pass_type == 0) {
        // Pass 1: Compute intermediate sums for grad_input
        // sum1 = sum(grad_output * gamma)
        // sum2 = sum(grad_output * gamma * x_norm)
        // These are stored in grad_gamma/grad_beta temporarily as scratch

        uint total_positions = batch_size * seq_len;
        if (gID >= total_positions) return;

        uint batch_idx = gID / seq_len;
        uint seq_idx = gID % seq_len;
        float mean = mean_vals[gID];
        float variance = var_vals[gID];
        float std_inv = 1.0 / sqrt(variance + eps);

        float sum1 = 0.0;
        float sum2 = 0.0;

        for (uint f = 0; f < features; f++) {
            uint idx = batch_idx * seq_len * features + seq_idx * features + f;
            float x = input_data[idx];
            float x_norm = (x - mean) * std_inv;
            float dy = grad_output[idx];
            float dy_gamma = dy * gamma[f];

            sum1 += dy_gamma;
            sum2 += dy_gamma * x_norm;
        }

        // Store sums in scratch space (reusing grad buffers temporarily)
        // We'll use first batch*seq entries of grad_gamma for sum1
        // and first batch*seq entries of grad_beta for sum2
        grad_gamma[gID] = sum1;
        grad_beta[gID] = sum2;

    } else if (pass_type == 1) {
        // Pass 2: Compute gradient w.r.t. input
        // dx = (1/N) * gamma/std * (N * dy - sum1 - x_norm * sum2)

        uint total_elements = batch_size * seq_len * features;
        if (gID >= total_elements) return;

        uint batch_idx = gID / (seq_len * features);
        uint remainder = gID % (seq_len * features);
        uint seq_idx = remainder / features;
        uint feat_idx = remainder % features;

        uint pos_idx = batch_idx * seq_len + seq_idx;
        float mean = mean_vals[pos_idx];
        float variance = var_vals[pos_idx];
        float std_inv = 1.0 / sqrt(variance + eps);

        float x = input_data[gID];
        float x_norm = (x - mean) * std_inv;
        float dy = grad_output[gID];
        float g = gamma[feat_idx];

        // Get precomputed sums
        float sum1 = grad_gamma[pos_idx];  // Sum of dy*gamma
        float sum2 = grad_beta[pos_idx];   // Sum of dy*gamma*x_norm

        float N = float(features);
        float dx = std_inv * g * (dy - (sum1 + x_norm * sum2) / N);

        grad_input[gID] = dx;

    } else if (pass_type == 2) {
        // Pass 3: Compute gradient w.r.t. gamma and beta
        // dgamma[f] = sum over batch,seq of: grad_output[b,s,f] * x_norm[b,s,f]
        // dbeta[f] = sum over batch,seq of: grad_output[b,s,f]

        // Each thread handles one feature
        if (gID >= features) return;

        float dgamma_sum = 0.0;
        float dbeta_sum = 0.0;

        for (uint b = 0; b < batch_size; b++) {
            for (uint s = 0; s < seq_len; s++) {
                uint pos_idx = b * seq_len + s;
                uint idx = b * seq_len * features + s * features + gID;

                float mean = mean_vals[pos_idx];
                float variance = var_vals[pos_idx];
                float std_inv = 1.0 / sqrt(variance + eps);

                float x = input_data[idx];
                float x_norm = (x - mean) * std_inv;
                float dy = grad_output[idx];

                dgamma_sum += dy * x_norm;
                dbeta_sum += dy;
            }
        }

        grad_gamma[gID] = dgamma_sum;
        grad_beta[gID] = dbeta_sum;
    }
}
