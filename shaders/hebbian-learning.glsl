#version 450

layout (local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

// Pre-synaptic activations (batch, time, pre_dim)
layout(set = 0, binding = 0) readonly buffer PreActivations {
    float pre[];
};

// Post-synaptic activations (batch, time, post_dim)
layout(set = 0, binding = 1) readonly buffer PostActivations {
    float post[];
};

// Weights to update (post_dim, pre_dim)
layout(set = 0, binding = 2) buffer Weights {
    float W[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint batch_size;
    uint time_steps;
    uint pre_dim;
    uint post_dim;
    float learning_rate;
    float weight_decay;     // Optional weight decay (L2 regularization)
};

void main() {
    // Each thread updates one weight: W[post_idx][pre_idx]
    uint post_idx = gl_GlobalInvocationID.y;
    uint pre_idx = gl_GlobalInvocationID.x;
    
    if (post_idx >= post_dim || pre_idx >= pre_dim) {
        return;
    }
    
    // Hebbian rule: ΔW = η * <pre * post>
    // Average over batch and time
    float correlation = 0.0;
    
    for (uint b = 0; b < batch_size; b++) {
        for (uint t = 0; t < time_steps; t++) {
            uint pre_idx_full = b * time_steps * pre_dim + t * pre_dim + pre_idx;
            uint post_idx_full = b * time_steps * post_dim + t * post_dim + post_idx;
            
            correlation += pre[pre_idx_full] * post[post_idx_full];
        }
    }
    
    // Average over batch and time
    correlation /= float(batch_size * time_steps);
    
    // Hebbian update with optional weight decay
    uint w_idx = post_idx * pre_dim + pre_idx;
    float current_w = W[w_idx];
    
    // ΔW = η * correlation - λ * W (weight decay)
    float delta_w = learning_rate * correlation - weight_decay * current_w;
    
    W[w_idx] = current_w + delta_w;
}
