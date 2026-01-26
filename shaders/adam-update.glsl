#version 450

// Adam Optimizer Update Shader
// Updates weights using Adam optimizer with accumulated gradients
// 
// Adam update rule:
// m = beta1 * m + (1 - beta1) * grad
// v = beta2 * v + (1 - beta2) * grad^2
// m_hat = m / (1 - beta1^t)
// v_hat = v / (1 - beta2^t)
// W = W - lr * m_hat / (sqrt(v_hat) + eps)

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Weights to update (capsule_dim * hidden_dim)
layout(set = 0, binding = 0) buffer Weights {
    float W[];
};

// Gradients (capsule_dim * hidden_dim)
layout(set = 0, binding = 1) buffer Gradients {
    float grad[];
};

// Adam first moment (m)
layout(set = 0, binding = 2) buffer Moment1 {
    float m[];
};

// Adam second moment (v)
layout(set = 0, binding = 3) buffer Moment2 {
    float v[];
};

layout(push_constant) uniform PushConsts {
    uint total_weights;
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    float beta1_t;  // beta1^t for bias correction
    float beta2_t;  // beta2^t for bias correction
    uint clear_grad; // 1 = clear gradients after update
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    
    if (idx >= total_weights) return;
    
    float g = grad[idx];
    
    // Update biased first moment estimate
    float m_new = beta1 * m[idx] + (1.0 - beta1) * g;
    m[idx] = m_new;
    
    // Update biased second raw moment estimate
    float v_new = beta2 * v[idx] + (1.0 - beta2) * g * g;
    v[idx] = v_new;
    
    // Compute bias-corrected estimates
    float m_hat = m_new / (1.0 - beta1_t);
    float v_hat = v_new / (1.0 - beta2_t);
    
    // Update weights
    W[idx] -= learning_rate * m_hat / (sqrt(v_hat) + epsilon);
    
    // Clear gradient for next iteration
    if (clear_grad == 1) {
        grad[idx] = 0.0;
    }
}
