#version 450

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Gradient w.r.t. softmax output (from upstream)
layout(set = 0, binding = 0) readonly buffer GradOutput {
    float grad_output[];
};

// Softmax output from forward pass
layout(set = 0, binding = 1) readonly buffer SoftmaxOutput {
    float softmax_out[];
};

// Output: gradient w.r.t. input (pre-softmax logits)
layout(set = 0, binding = 2) buffer GradInput {
    float grad_input[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint batch_size;
    uint seq_len;      // Number of rows
    uint num_classes;  // Number of columns (softmax dimension)
};

void main() {
    uint gID = gl_GlobalInvocationID.x;

    // Each thread handles one row (one softmax application)
    uint total_rows = batch_size * seq_len;
    if (gID >= total_rows) return;

    uint row_offset = gID * num_classes;

    // Compute sum(grad_output * softmax_output) for this row
    float dot_product = 0.0;
    for (uint c = 0; c < num_classes; c++) {
        uint idx = row_offset + c;
        dot_product += grad_output[idx] * softmax_out[idx];
    }

    // Compute grad_input = softmax_output * (grad_output - dot_product)
    for (uint c = 0; c < num_classes; c++) {
        uint idx = row_offset + c;
        float s = softmax_out[idx];
        float dy = grad_output[idx];
        grad_input[idx] = s * (dy - dot_product);
    }
}
