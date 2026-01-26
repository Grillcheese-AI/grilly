#version 450
#extension GL_EXT_shader_atomic_float : require

layout (local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

// Gradient w.r.t. output (batch * seq, output_dim)
layout(set = 0, binding = 0) readonly buffer GradOutput {
    float grad_output[];
};

// Input activations (batch * seq, input_dim)
layout(set = 0, binding = 1) readonly buffer Input {
    float input_data[];
};

// Weights (output_dim, input_dim)
layout(set = 0, binding = 2) readonly buffer Weights {
    float W[];
};

// Gradient w.r.t. input (batch * seq, input_dim)
layout(set = 0, binding = 3) buffer GradInput {
    float grad_input[];
};

// Gradient w.r.t. weights (output_dim, input_dim) - accumulated
layout(set = 0, binding = 4) buffer GradWeights {
    float grad_W[];
};

// Gradient w.r.t. bias (output_dim) - accumulated
layout(set = 0, binding = 5) buffer GradBias {
    float grad_b[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint batch_seq;
    uint input_dim;
    uint output_dim;
    uint pass_type;      // 0 = grad_input, 1 = grad_weights, 2 = grad_bias
};

void main() {
    uint row = gl_GlobalInvocationID.y;
    uint col = gl_GlobalInvocationID.x;
    
    if (pass_type == 0) {
        // Compute gradient w.r.t. input: grad_input = grad_output @ W
        if (row >= batch_seq || col >= input_dim) return;
        
        float sum = 0.0;
        for (uint k = 0; k < output_dim; k++) {
            uint grad_out_idx = row * output_dim + k;
            uint w_idx = k * input_dim + col;
            sum += grad_output[grad_out_idx] * W[w_idx];
        }
        
        uint grad_in_idx = row * input_dim + col;
        grad_input[grad_in_idx] = sum;
        
    } else if (pass_type == 1) {
        // Compute gradient w.r.t. weights: grad_W = grad_output^T @ input
        if (row >= output_dim || col >= input_dim) return;
        
        float sum = 0.0;
        for (uint b = 0; b < batch_seq; b++) {
            uint grad_out_idx = b * output_dim + row;
            uint input_idx = b * input_dim + col;
            sum += grad_output[grad_out_idx] * input_data[input_idx];
        }
        
        uint grad_w_idx = row * input_dim + col;
        atomicAdd(grad_W[grad_w_idx], sum);
        
    } else if (pass_type == 2) {
        // Compute gradient w.r.t. bias: grad_b = sum(grad_output, dim=0)
        uint out_idx = gl_GlobalInvocationID.x;
        if (out_idx >= output_dim) return;
        
        float sum = 0.0;
        for (uint b = 0; b < batch_seq; b++) {
            uint grad_out_idx = b * output_dim + out_idx;
            sum += grad_output[grad_out_idx];
        }
        
        atomicAdd(grad_b[out_idx], sum);
    }
}
