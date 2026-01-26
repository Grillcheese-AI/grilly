#version 450

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Input features (batch, n_features)
layout(set = 0, binding = 0) readonly buffer InputFeatures {
    float x[];
};

// Weights (n_features)
layout(set = 0, binding = 1) readonly buffer Weights {
    float w[];
};

// Bias (single value)
layout(set = 0, binding = 2) readonly buffer Bias {
    float bias[];
};

// Predictions output (batch)
layout(set = 0, binding = 3) buffer Predictions {
    float y_pred[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint batch_size;
    uint n_features;
};

void main() {
    uint batch_idx = gl_GlobalInvocationID.x;
    
    if (batch_idx >= batch_size) {
        return;
    }
    
    // Linear prediction: y = w · x + b
    float prediction = 0.0;
    
    for (uint i = 0; i < n_features; i++) {
        uint x_idx = batch_idx * n_features + i;
        prediction += w[i] * x[x_idx];
    }
    
    prediction += bias[0];
    
    y_pred[batch_idx] = prediction;
}
