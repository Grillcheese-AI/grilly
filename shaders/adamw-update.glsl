#version 450

// AdamW optimizer update shader (decoupled weight decay).
// m = beta1 * m + (1 - beta1) * g
// v = beta2 * v + (1 - beta2) * g^2
// m_hat = m / (1 - beta1^t)
// v_hat = v / (1 - beta2^t)
// w = w * (1 - lr * wd) - lr * m_hat / (sqrt(v_hat) + eps)

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer Weights {
    float W[];
};

layout(set = 0, binding = 1) buffer Gradients {
    float grad[];
};

layout(set = 0, binding = 2) buffer Moment1 {
    float m[];
};

layout(set = 0, binding = 3) buffer Moment2 {
    float v[];
};

layout(push_constant) uniform PushConsts {
    uint total_weights;
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    float weight_decay;
    float beta1_t;
    float beta2_t;
    uint clear_grad;
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= total_weights) {
        return;
    }

    float g = grad[idx];
    float m_new = beta1 * m[idx] + (1.0 - beta1) * g;
    float v_new = beta2 * v[idx] + (1.0 - beta2) * g * g;
    m[idx] = m_new;
    v[idx] = v_new;

    float m_hat = m_new / max(1e-12, (1.0 - beta1_t));
    float v_hat = v_new / max(1e-12, (1.0 - beta2_t));

    float decayed = W[idx] * (1.0 - learning_rate * weight_decay);
    W[idx] = decayed - learning_rate * m_hat / (sqrt(max(v_hat, 0.0)) + epsilon);

    if (clear_grad != 0u) {
        grad[idx] = 0.0;
    }
}
