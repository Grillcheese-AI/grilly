#version 450

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// MLP pathway output (batch, seq, features)
layout(set = 0, binding = 0) readonly buffer MLPOutput {
    float mlp[];
};

// SNN pathway output (batch, seq, features)
layout(set = 0, binding = 1) readonly buffer SNNOutput {
    float snn[];
};

// Blended output (batch, seq, features)
layout(set = 0, binding = 2) buffer BlendedOutput {
    float output_data[];
};

// Optional: learnable gate parameter (single value or per-feature)
layout(set = 0, binding = 3) readonly buffer GateParams {
    float gate_raw[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint total_elements;   // batch * seq * features
    float fixed_gate;      // Fixed gate value (0-1), used if use_learnable_gate == 0
    uint use_learnable_gate; // 1 = use gate_raw buffer, 0 = use fixed_gate
    uint gate_is_scalar;   // 1 = single gate value, 0 = per-feature gates
    uint num_features;     // Feature dimension (for per-feature gating)
};

// Sigmoid activation
float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

void main() {
    uint gID = gl_GlobalInvocationID.x;
    
    if (gID >= total_elements) {
        return;
    }
    
    // Read inputs
    float mlp_val = mlp[gID];
    float snn_val = snn[gID];
    
    // Determine gate value
    float gate = fixed_gate;
    
    if (use_learnable_gate == 1) {
        if (gate_is_scalar == 1) {
            // Single learnable gate for all features
            gate = sigmoid(gate_raw[0]);
        } else {
            // Per-feature learnable gate
            uint feat_idx = gID % num_features;
            gate = sigmoid(gate_raw[feat_idx]);
        }
    }
    
    // Blend: output = (1 - gate) * mlp + gate * snn
    float result = (1.0 - gate) * mlp_val + gate * snn_val;
    
    output_data[gID] = result;
}
