#version 450

// Adam optimizer for weight updates
// Implements: m = β1*m + (1-β1)*g, v = β2*v + (1-β2)*g², 
//             m_hat = m/(1-β1^t), v_hat = v/(1-β2^t),
//             w = w - lr * m_hat / (sqrt(v_hat) + ε)

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Gradients (flattened)
layout(set = 0, binding = 0) readonly buffer Gradients {
    float grad[];
};

// Weights to update (flattened)
layout(set = 0, binding = 1) buffer Weights {
    float W[];
};

// First moment (m) - same size as weights
layout(set = 0, binding = 2) buffer Moment1 {
    float m[];
};

// Second moment (v) - same size as weights  
layout(set = 0, binding = 3) buffer Moment2 {
    float v[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint total_weights;
    float learning_rate;
    float beta1;           // First moment decay (typically 0.9)
    float beta2;           // Second moment decay (typically 0.999)
    float epsilon;         // Numerical stability (typically 1e-8)
    float weight_decay;    // L2 regularization
    uint timestep;         // Current timestep for bias correction
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    
    if (idx >= total_weights) {
        return;
    }
    
    float g = grad[idx];
    float w = W[idx];
    
    // Add weight decay to gradient (L2 regularization)
    g = g + weight_decay * w;
    
    // Update first moment (momentum)
    float m_new = beta1 * m[idx] + (1.0 - beta1) * g;
    m[idx] = m_new;
    
    // Update second moment (RMSprop-like)
    float v_new = beta2 * v[idx] + (1.0 - beta2) * g * g;
    v[idx] = v_new;
    
    // Bias correction
    float beta1_t = pow(beta1, float(timestep));
    float beta2_t = pow(beta2, float(timestep));
    float m_hat = m_new / (1.0 - beta1_t);
    float v_hat = v_new / (1.0 - beta2_t);
    
    // Update weights
    W[idx] = w - learning_rate * m_hat / (sqrt(v_hat) + epsilon);
}

