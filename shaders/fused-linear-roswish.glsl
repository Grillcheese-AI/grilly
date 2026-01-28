#version 450

/*
 * Fused Linear + RoSwish Shader
 *
 * Combines linear transformation and RoSwish activation in a single pass.
 * RoSwish: (x + α) * sigmoid(β * x) - 0.5 * α
 *
 * Learnable activation with adaptive gating for general neural networks.
 */

layout (local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

// Input features (batch * seq, input_dim)
layout(set = 0, binding = 0) readonly buffer Input {
    float input_data[];
};

// Weight matrix (output_dim, input_dim)
layout(set = 0, binding = 1) readonly buffer Weights {
    float W[];
};

// Bias vector (output_dim)
layout(set = 0, binding = 2) readonly buffer Bias {
    float b[];
};

// Output (batch * seq, output_dim)
layout(set = 0, binding = 3) buffer Output {
    float output_data[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint batch_seq;       // batch_size * seq_len
    uint input_dim;
    uint output_dim;
    uint has_bias;        // 1 if bias exists, 0 otherwise
    float alpha;          // RoSwish rotation parameter
    float beta;           // RoSwish gating parameter
};

// Inline RoSwish with numerically stable sigmoid
float roswish(float x, float alpha, float beta) {
    float beta_x = beta * x;
    float sigmoid_bx;

    if (beta_x >= 0.0) {
        sigmoid_bx = 1.0 / (1.0 + exp(-beta_x));
    } else {
        float exp_bx = exp(beta_x);
        sigmoid_bx = exp_bx / (1.0 + exp_bx);
    }

    return (x + alpha) * sigmoid_bx - 0.5 * alpha;
}

void main() {
    uint row = gl_GlobalInvocationID.y;  // Sample index
    uint col = gl_GlobalInvocationID.x;  // Output feature index

    if (row >= batch_seq || col >= output_dim) {
        return;
    }

    // Step 1: Linear transformation
    float sum = 0.0;

    for (uint k = 0; k < input_dim; k++) {
        uint input_idx = row * input_dim + k;
        uint weight_idx = col * input_dim + k;
        sum += input_data[input_idx] * W[weight_idx];
    }

    // Add bias if present
    if (has_bias == 1) {
        sum += b[col];
    }

    // Step 2: Apply RoSwish activation (fused)
    float roswish_out = roswish(sum, alpha, beta);

    // Write final result
    uint out_idx = row * output_dim + col;
    output_data[out_idx] = roswish_out;
}
