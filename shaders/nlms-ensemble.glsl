#version 450

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Predictions from multiple experts (batch, num_experts)
layout(set = 0, binding = 0) readonly buffer ExpertPredictions {
    float predictions[];
};

// Expert RMSEs (num_experts)
layout(set = 0, binding = 1) readonly buffer ExpertRMSEs {
    float rmses[];
};

// Gating weights output (batch, num_experts)
layout(set = 0, binding = 2) buffer GatingWeights {
    float gates[];
};

// Combined predictions (batch)
layout(set = 0, binding = 3) buffer CombinedPredictions {
    float combined[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint batch_size;
    uint num_experts;
    float temperature;    // Softmax temperature for gating (e.g., 1.0)
    uint gating_mode;     // 0 = performance-based, 1 = uniform
};

void main() {
    uint batch_idx = gl_GlobalInvocationID.x;
    
    if (batch_idx >= batch_size) {
        return;
    }
    
    // Compute gating weights based on expert performance
    float gate_logits[64];  // Assuming num_experts <= 64
    float max_logit = -1e10;
    
    for (uint e = 0; e < num_experts && e < 64; e++) {
        if (gating_mode == 0) {
            // Performance-based gating: better experts (lower RMSE) get higher weight
            // logit = -RMSE / temperature
            gate_logits[e] = -rmses[e] / temperature;
        } else {
            // Uniform gating
            gate_logits[e] = 0.0;
        }
        max_logit = max(max_logit, gate_logits[e]);
    }
    
    // Softmax normalization
    float sum_exp = 0.0;
    for (uint e = 0; e < num_experts && e < 64; e++) {
        float exp_val = exp(gate_logits[e] - max_logit);
        gate_logits[e] = exp_val;
        sum_exp += exp_val;
    }
    
    // Normalize and compute weighted prediction
    float combined_pred = 0.0;
    
    for (uint e = 0; e < num_experts && e < 64; e++) {
        float gate = gate_logits[e] / sum_exp;
        uint pred_idx = batch_idx * num_experts + e;
        uint gate_idx = batch_idx * num_experts + e;
        
        gates[gate_idx] = gate;
        combined_pred += gate * predictions[pred_idx];
    }
    
    combined[batch_idx] = combined_pred;
}
