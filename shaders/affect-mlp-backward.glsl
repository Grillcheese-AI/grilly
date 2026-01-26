#version 450

// Backpropagation for 3-layer affect MLP
// Computes gradients for all layers

layout (local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

// Input embeddings (batch, embedding_dim) - for gradient computation
layout(set = 0, binding = 0) readonly buffer Input {
    float input_data[];
};

// Hidden1 activations (batch, hidden1_dim) - saved from forward pass
layout(set = 0, binding = 1) readonly buffer Hidden1 {
    float hidden1[];
};

// Hidden1 pre-activation (batch, hidden1_dim) - for LeakyReLU gradient
layout(set = 0, binding = 2) readonly buffer Hidden1Pre {
    float hidden1_pre[];
};

// Hidden2 activations (batch, hidden2_dim) - saved from forward pass
layout(set = 0, binding = 3) readonly buffer Hidden2 {
    float hidden2[];
};

// Hidden2 pre-activation (batch, hidden2_dim) - for LeakyReLU gradient
layout(set = 0, binding = 4) readonly buffer Hidden2Pre {
    float hidden2_pre[];
};

// Output predictions (batch, 2)
layout(set = 0, binding = 5) readonly buffer Predictions {
    float predictions[];
};

// Targets (batch, 2) - [valence, arousal]
layout(set = 0, binding = 6) readonly buffer Targets {
    float targets[];
};

// Layer 3 weights (2, hidden2_dim) - for backprop
layout(set = 0, binding = 7) readonly buffer W3 {
    float w3[];
};

// Layer 2 weights (hidden2_dim, hidden1_dim) - for backprop
layout(set = 0, binding = 8) readonly buffer W2 {
    float w2[];
};

// Gradients for W1 (hidden1_dim, embedding_dim)
layout(set = 0, binding = 9) buffer GradW1 {
    float grad_w1[];
};

// Gradients for W2 (hidden2_dim, hidden1_dim)
layout(set = 0, binding = 10) buffer GradW2 {
    float grad_w2[];
};

// Gradients for W3 (2, hidden2_dim)
layout(set = 0, binding = 11) buffer GradW3 {
    float grad_w3[];
};

// Gradients for b1 (hidden1_dim)
layout(set = 0, binding = 12) buffer GradB1 {
    float grad_b1[];
};

// Gradients for b2 (hidden2_dim)
layout(set = 0, binding = 13) buffer GradB2 {
    float grad_b2[];
};

// Gradients for b3 (2)
layout(set = 0, binding = 14) buffer GradB3 {
    float grad_b3[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint batch_size;
    uint embedding_dim;
    uint hidden1_dim;
    uint hidden2_dim;
    float leaky_slope;
    uint pass_type;  // 0 = grad_w3/b3, 1 = grad_w2/b2, 2 = grad_w1/b1
};

// LeakyReLU derivative
float leaky_relu_grad(float pre_act, float slope) {
    return pre_act > 0.0 ? 1.0 : slope;
}

// Shared memory for batch accumulation
shared float shared_sum[256];

void main() {
    uint x = gl_GlobalInvocationID.x;
    uint y = gl_GlobalInvocationID.y;
    
    if (pass_type == 0) {
        // Pass 0: Compute gradients for W3 and b3
        // W3: (2, hidden2_dim), each thread handles one weight
        uint out_idx = y;  // 0 or 1
        uint h2_idx = x;
        
        if (out_idx >= 2 || h2_idx >= hidden2_dim) return;
        
        float grad_sum = 0.0;
        float bias_grad = 0.0;
        
        for (uint b = 0; b < batch_size; b++) {
            // d_output = predictions - targets (MSE gradient)
            float d_out = (predictions[b * 2 + out_idx] - targets[b * 2 + out_idx]) * 2.0 / float(batch_size);
            
            // Gradient for W3
            float h2_val = hidden2[b * hidden2_dim + h2_idx];
            grad_sum += d_out * h2_val;
            
            // Bias gradient (only accumulate once per output)
            if (h2_idx == 0) {
                bias_grad += d_out;
            }
        }
        
        uint w3_idx = out_idx * hidden2_dim + h2_idx;
        grad_w3[w3_idx] = grad_sum;
        
        if (h2_idx == 0) {
            grad_b3[out_idx] = bias_grad;
        }
        
    } else if (pass_type == 1) {
        // Pass 1: Compute gradients for W2 and b2
        uint h2_idx = y;
        uint h1_idx = x;
        
        if (h2_idx >= hidden2_dim || h1_idx >= hidden1_dim) return;
        
        float grad_sum = 0.0;
        float bias_grad = 0.0;
        
        for (uint b = 0; b < batch_size; b++) {
            // Backprop from output layer
            float d_h2 = 0.0;
            for (uint o = 0; o < 2; o++) {
                float d_out = (predictions[b * 2 + o] - targets[b * 2 + o]) * 2.0 / float(batch_size);
                d_h2 += d_out * w3[o * hidden2_dim + h2_idx];
            }
            
            // LeakyReLU gradient
            float h2_pre = hidden2_pre[b * hidden2_dim + h2_idx];
            d_h2 *= leaky_relu_grad(h2_pre, leaky_slope);
            
            // Gradient for W2
            float h1_val = hidden1[b * hidden1_dim + h1_idx];
            grad_sum += d_h2 * h1_val;
            
            if (h1_idx == 0) {
                bias_grad += d_h2;
            }
        }
        
        uint w2_idx = h2_idx * hidden1_dim + h1_idx;
        grad_w2[w2_idx] = grad_sum;
        
        if (h1_idx == 0) {
            grad_b2[h2_idx] = bias_grad;
        }
        
    } else if (pass_type == 2) {
        // Pass 2: Compute gradients for W1 and b1
        uint h1_idx = y;
        uint e_idx = x;
        
        if (h1_idx >= hidden1_dim || e_idx >= embedding_dim) return;
        
        float grad_sum = 0.0;
        float bias_grad = 0.0;
        
        for (uint b = 0; b < batch_size; b++) {
            // Backprop from layer 2
            float d_h1 = 0.0;
            
            for (uint h2 = 0; h2 < hidden2_dim; h2++) {
                // Compute d_h2 first
                float d_h2 = 0.0;
                for (uint o = 0; o < 2; o++) {
                    float d_out = (predictions[b * 2 + o] - targets[b * 2 + o]) * 2.0 / float(batch_size);
                    d_h2 += d_out * w3[o * hidden2_dim + h2];
                }
                float h2_pre = hidden2_pre[b * hidden2_dim + h2];
                d_h2 *= leaky_relu_grad(h2_pre, leaky_slope);
                
                // Accumulate to d_h1
                d_h1 += d_h2 * w2[h2 * hidden1_dim + h1_idx];
            }
            
            // Residual connection gradient (if dims match)
            if (hidden1_dim == hidden2_dim) {
                float d_h2 = 0.0;
                for (uint o = 0; o < 2; o++) {
                    float d_out = (predictions[b * 2 + o] - targets[b * 2 + o]) * 2.0 / float(batch_size);
                    d_h2 += d_out * w3[o * hidden2_dim + h1_idx];
                }
                float h2_pre = hidden2_pre[b * hidden2_dim + h1_idx];
                d_h1 += d_h2 * leaky_relu_grad(h2_pre, leaky_slope);
            }
            
            // LeakyReLU gradient for layer 1
            float h1_pre = hidden1_pre[b * hidden1_dim + h1_idx];
            d_h1 *= leaky_relu_grad(h1_pre, leaky_slope);
            
            // Gradient for W1
            float input_val = input_data[b * embedding_dim + e_idx];
            grad_sum += d_h1 * input_val;
            
            if (e_idx == 0) {
                bias_grad += d_h1;
            }
        }
        
        uint w1_idx = h1_idx * embedding_dim + e_idx;
        grad_w1[w1_idx] = grad_sum;
        
        if (e_idx == 0) {
            grad_b1[h1_idx] = bias_grad;
        }
    }
}

