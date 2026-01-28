#version 450

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Gradient from next layer (shape: batch * hidden_dim)
layout(set = 0, binding = 0) readonly buffer GradOutput {
    float grad_output[];
};

// Input tensor (shape: batch * 2*hidden_dim, for computing derivative)
layout(set = 0, binding = 1) readonly buffer Input {
    float input_data[];
};

// Gradient w.r.t. input (shape: batch * 2*hidden_dim)
layout(set = 0, binding = 2) buffer GradInput {
    float grad_input[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint output_elements;  // batch_size * hidden_dim
    uint hidden_dim;       // Size of each split
};

void main() {
    uint gID = gl_GlobalInvocationID.x;

    if (gID >= output_elements) {
        return;
    }

    // Calculate which batch and position within batch
    uint batch_idx = gID / hidden_dim;
    uint pos_in_batch = gID % hidden_dim;

    // Input layout: [batch][x1: hidden_dim][x2: hidden_dim]
    uint x1_idx = batch_idx * (hidden_dim * 2) + pos_in_batch;
    uint x2_idx = batch_idx * (hidden_dim * 2) + hidden_dim + pos_in_batch;

    float x1 = input_data[x1_idx];
    float x2 = input_data[x2_idx];
    float grad_out = grad_output[gID];

    // Forward: output = x1 * silu(x2) = x1 * x2 * sigmoid(x2)
    // d/dx1 = silu(x2)
    // d/dx2 = x1 * d/dx2(silu(x2)) = x1 * (sigmoid(x2) + x2 * sigmoid(x2) * (1 - sigmoid(x2)))

    // Compute sigmoid(x2) numerically stable
    float sigmoid_x2;
    if (x2 >= 0.0) {
        sigmoid_x2 = 1.0 / (1.0 + exp(-x2));
    } else {
        float exp_x2 = exp(x2);
        sigmoid_x2 = exp_x2 / (1.0 + exp_x2);
    }

    float silu_x2 = x2 * sigmoid_x2;

    // Gradient w.r.t. x1: grad_out * silu(x2)
    grad_input[x1_idx] = grad_out * silu_x2;

    // Gradient w.r.t. x2: grad_out * x1 * d/dx2(silu(x2))
    // d/dx2(silu(x2)) = sigmoid(x2) + x2 * sigmoid(x2) * (1 - sigmoid(x2))
    float silu_derivative = sigmoid_x2 * (1.0 + x2 * (1.0 - sigmoid_x2));
    grad_input[x2_idx] = grad_out * x1 * silu_derivative;
}
