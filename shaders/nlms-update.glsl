#version 450

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Input features (n_features)
layout(set = 0, binding = 0) readonly buffer InputFeatures {
    float x[];
};

// Predicted value (single)
layout(set = 0, binding = 1) readonly buffer Prediction {
    float y_pred[];
};

// True value (single)
layout(set = 0, binding = 2) readonly buffer TrueValue {
    float y_true[];
};

// Weights (n_features) - read and write
layout(set = 0, binding = 3) buffer Weights {
    float w[];
};

// Bias (single value) - read and write
layout(set = 0, binding = 4) buffer Bias {
    float bias[];
};

// Learning rate (single value) - read and write
layout(set = 0, binding = 5) buffer LearningRate {
    float mu[];
};

// Error output (single value)
layout(set = 0, binding = 6) buffer Error {
    float error[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint n_features;
    float mu_decay;          // Learning rate decay (e.g., 0.99995)
    float mu_min;            // Minimum learning rate (e.g., 0.1)
    float bias_lr_scale;     // Bias learning rate scale (e.g., 0.1)
    float epsilon;           // Normalization constant (e.g., 1e-6)
};

void main() {
    uint gID = gl_GlobalInvocationID.x;
    
    if (gID == 0) {
        // Thread 0 computes error and updates bias
        float err = y_true[0] - y_pred[0];
        error[0] = err;
        
        // Update bias: b = b + μ * error * scale
        float current_mu = mu[0];
        bias[0] += current_mu * err * bias_lr_scale;
        
        // Decay learning rate
        if (current_mu > mu_min) {
            mu[0] = current_mu * mu_decay;
        }
    }
    
    // All threads update weights in parallel
    if (gID < n_features) {
        // Compute ||x||^2 (each thread computes partial, then we'd need reduction)
        // For simplicity, approximate with sum approach
        float norm_sq = 0.0;
        for (uint i = 0; i < n_features; i++) {
            norm_sq += x[i] * x[i];
        }
        norm_sq += epsilon;
        
        // NLMS update: w[i] = w[i] + (μ * error * x[i]) / ||x||^2
        float err = y_true[0] - y_pred[0];
        float current_mu = mu[0];
        float step = (current_mu * err) / norm_sq;
        
        w[gID] += step * x[gID];
    }
}
