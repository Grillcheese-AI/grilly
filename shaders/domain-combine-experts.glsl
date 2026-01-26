#version 450

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Expert outputs (batch, num_experts, dim)
layout(set = 0, binding = 0) readonly buffer ExpertOutputs {
    float expert_out[];
};

// Routing weights (batch, num_experts)
layout(set = 0, binding = 1) readonly buffer RoutingWeights {
    float weights[];
};

// Combined output (batch, dim)
layout(set = 0, binding = 2) buffer CombinedOutput {
    float output_data[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint batch_size;
    uint num_experts;
    uint dim;
    uint normalize_weights;  // 1 = normalize routing weights, 0 = use as-is
};

void main() {
    uint gID = gl_GlobalInvocationID.x;
    uint total_elements = batch_size * dim;
    
    if (gID >= total_elements) {
        return;
    }
    
    uint batch_idx = gID / dim;
    uint dim_idx = gID % dim;
    
    // Normalize routing weights if needed
    float weight_sum = 0.0;
    if (normalize_weights == 1) {
        for (uint e = 0; e < num_experts; e++) {
            uint w_idx = batch_idx * num_experts + e;
            weight_sum += weights[w_idx];
        }
    } else {
        weight_sum = 1.0;
    }
    
    // Weighted sum of expert outputs
    float combined = 0.0;
    
    for (uint e = 0; e < num_experts; e++) {
        uint w_idx = batch_idx * num_experts + e;
        uint expert_idx = batch_idx * num_experts * dim + e * dim + dim_idx;
        
        float w = weights[w_idx] / weight_sum;
        combined += w * expert_out[expert_idx];
    }
    
    output_data[gID] = combined;
}
