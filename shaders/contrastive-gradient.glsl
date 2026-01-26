#version 450

// Contrastive Gradient Computation Shader
// Computes gradients for capsule projection matrix W
// 
// For triplet loss L = max(0, d(a,p) - d(a,n) + margin):
// dL/dW = dL/d_capsule * d_capsule/dW
// 
// Where:
// d_capsule/dW[i,j] = input[j] (for output dimension i)

layout (local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

// Anchor embeddings (batch, hidden_dim)
layout(set = 0, binding = 0) readonly buffer Anchors {
    float anchors[];
};

// Positive embeddings (batch, hidden_dim)  
layout(set = 0, binding = 1) readonly buffer Positives {
    float positives[];
};

// Negative embeddings (num_negatives, hidden_dim)
layout(set = 0, binding = 2) readonly buffer Negatives {
    float negatives[];
};

// Anchor capsules (batch, capsule_dim)
layout(set = 0, binding = 3) readonly buffer AnchorCaps {
    float anchor_caps[];
};

// Positive capsules (batch, capsule_dim)
layout(set = 0, binding = 4) readonly buffer PositiveCaps {
    float positive_caps[];
};

// Negative capsules (num_negatives, capsule_dim)
layout(set = 0, binding = 5) readonly buffer NegativeCaps {
    float negative_caps[];
};

// Hardest negative indices (batch,)
layout(set = 0, binding = 6) readonly buffer HardestIdx {
    int hardest_idx[];
};

// Losses (batch,) - used to mask zero-loss samples
layout(set = 0, binding = 7) readonly buffer Losses {
    float losses[];
};

// Gradient accumulator for W (capsule_dim, hidden_dim)
layout(set = 0, binding = 8) buffer GradW {
    float grad_W[];
};

layout(push_constant) uniform PushConsts {
    uint batch_size;
    uint hidden_dim;
    uint capsule_dim;
    float learning_rate;
};

void main() {
    uint cap_idx = gl_GlobalInvocationID.x;  // Output dimension
    uint hidden_idx = gl_GlobalInvocationID.y;  // Input dimension
    
    if (cap_idx >= capsule_dim || hidden_idx >= hidden_dim) return;
    
    float grad_sum = 0.0;
    
    // Accumulate gradients across batch
    for (uint b = 0; b < batch_size; b++) {
        // Skip if loss is zero (triplet constraint satisfied)
        if (losses[b] <= 0.0) continue;
        
        int hard_neg = hardest_idx[b];
        
        // Get capsule vectors
        float a_cap = anchor_caps[b * capsule_dim + cap_idx];
        float p_cap = positive_caps[b * capsule_dim + cap_idx];
        float n_cap = negative_caps[hard_neg * capsule_dim + cap_idx];
        
        // Get input vectors
        float a_in = anchors[b * hidden_dim + hidden_idx];
        float p_in = positives[b * hidden_dim + hidden_idx];
        float n_in = negatives[hard_neg * hidden_dim + hidden_idx];
        
        // Gradient through positive term: -d(cos_sim)/dW
        // d(cos_sim(a,p))/dW[i,j] ≈ (p[i] * a_in + a[i] * p_in) / (||a|| * ||p||)
        // Simplified gradient (normalized vectors):
        float grad_pos = -(p_cap * a_in + a_cap * p_in);
        
        // Gradient through negative term: +d(cos_sim)/dW  
        float grad_neg = (n_cap * a_in + a_cap * n_in);
        
        grad_sum += (grad_pos + grad_neg);
    }
    
    // Average and scale by learning rate
    grad_sum = grad_sum / float(batch_size) * learning_rate;
    
    // Accumulate to gradient buffer (atomic add would be better but not available)
    uint w_idx = cap_idx * hidden_dim + hidden_idx;
    grad_W[w_idx] += grad_sum;
}
