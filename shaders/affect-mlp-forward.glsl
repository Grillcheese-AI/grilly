#version 450

// Fused 3-layer MLP for affect prediction with LeakyReLU
// Architecture: embedding -> hidden1 -> hidden2 -> [valence, arousal]
// Includes residual connection from hidden1 to hidden2

layout (local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

// Input embeddings (batch, embedding_dim)
layout(set = 0, binding = 0) readonly buffer Input {
    float input_data[];
};

// Layer 1 weights (hidden1_dim, embedding_dim)
layout(set = 0, binding = 1) readonly buffer W1 {
    float w1[];
};

// Layer 1 bias (hidden1_dim)
layout(set = 0, binding = 2) readonly buffer B1 {
    float b1[];
};

// Layer 2 weights (hidden2_dim, hidden1_dim)
layout(set = 0, binding = 3) readonly buffer W2 {
    float w2[];
};

// Layer 2 bias (hidden2_dim)
layout(set = 0, binding = 4) readonly buffer B2 {
    float b2[];
};

// Output projection weights (2, hidden2_dim) - valence and arousal
layout(set = 0, binding = 5) readonly buffer W3 {
    float w3[];
};

// Output bias (2)
layout(set = 0, binding = 6) readonly buffer B3 {
    float b3[];
};

// Hidden1 activations for backprop (batch, hidden1_dim)
layout(set = 0, binding = 7) buffer Hidden1 {
    float hidden1[];
};

// Hidden2 activations for backprop (batch, hidden2_dim)
layout(set = 0, binding = 8) buffer Hidden2 {
    float hidden2[];
};

// Output predictions (batch, 2) - [valence, arousal]
layout(set = 0, binding = 9) buffer Output {
    float output_data[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint batch_size;
    uint embedding_dim;
    uint hidden1_dim;
    uint hidden2_dim;
    float leaky_slope;     // Negative slope for LeakyReLU (e.g., 0.01)
    uint apply_output_act; // 1 = apply tanh/sigmoid, 0 = linear
    float dropout_rate;    // Dropout rate (0 during inference)
    uint seed;             // Random seed for dropout
};

// LeakyReLU activation
float leaky_relu(float x, float slope) {
    return x > 0.0 ? x : slope * x;
}

// Simple hash for pseudo-random dropout
float hash(uint seed, uint idx) {
    uint h = seed ^ idx;
    h = h * 2654435761u;
    h = h ^ (h >> 16);
    return float(h & 0xFFFFu) / 65535.0;
}

void main() {
    uint batch_idx = gl_GlobalInvocationID.x;
    
    if (batch_idx >= batch_size) {
        return;
    }
    
    // ===== Layer 1: embedding -> hidden1 =====
    for (uint h1 = 0; h1 < hidden1_dim; h1++) {
        float sum = b1[h1];
        for (uint e = 0; e < embedding_dim; e++) {
            uint in_idx = batch_idx * embedding_dim + e;
            uint w_idx = h1 * embedding_dim + e;
            sum += input_data[in_idx] * w1[w_idx];
        }
        // LeakyReLU activation
        float activated = leaky_relu(sum, leaky_slope);
        
        // Dropout (training only)
        if (dropout_rate > 0.0) {
            float r = hash(seed, batch_idx * hidden1_dim + h1);
            if (r < dropout_rate) {
                activated = 0.0;
            } else {
                activated = activated / (1.0 - dropout_rate); // Scale
            }
        }
        
        hidden1[batch_idx * hidden1_dim + h1] = activated;
    }
    
    // ===== Layer 2: hidden1 -> hidden2 (with residual if dims match) =====
    for (uint h2 = 0; h2 < hidden2_dim; h2++) {
        float sum = b2[h2];
        for (uint h1 = 0; h1 < hidden1_dim; h1++) {
            uint h1_idx = batch_idx * hidden1_dim + h1;
            uint w_idx = h2 * hidden1_dim + h1;
            sum += hidden1[h1_idx] * w2[w_idx];
        }
        
        // Residual connection (if hidden1_dim == hidden2_dim)
        if (hidden1_dim == hidden2_dim) {
            sum += hidden1[batch_idx * hidden1_dim + h2];
        }
        
        // LeakyReLU activation
        float activated = leaky_relu(sum, leaky_slope);
        
        // Dropout
        if (dropout_rate > 0.0) {
            float r = hash(seed + 1, batch_idx * hidden2_dim + h2);
            if (r < dropout_rate) {
                activated = 0.0;
            } else {
                activated = activated / (1.0 - dropout_rate);
            }
        }
        
        hidden2[batch_idx * hidden2_dim + h2] = activated;
    }
    
    // ===== Layer 3: hidden2 -> output [valence, arousal] =====
    for (uint o = 0; o < 2; o++) {
        float sum = b3[o];
        for (uint h2 = 0; h2 < hidden2_dim; h2++) {
            uint h2_idx = batch_idx * hidden2_dim + h2;
            uint w_idx = o * hidden2_dim + h2;
            sum += hidden2[h2_idx] * w3[w_idx];
        }
        
        // Apply output activations if requested
        if (apply_output_act == 1) {
            if (o == 0) {
                // Valence: tanh [-1, 1]
                sum = tanh(sum);
            } else {
                // Arousal: sigmoid [0, 1]
                sum = 1.0 / (1.0 + exp(-clamp(sum, -10.0, 10.0)));
            }
        }
        
        output_data[batch_idx * 2 + o] = sum;
    }
}

