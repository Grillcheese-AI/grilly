#version 450

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Domain probabilities (batch, num_domains)
layout(set = 0, binding = 0) readonly buffer DomainProbs {
    float probs[];
};

// Expert weights for each domain (num_domains, num_experts)
layout(set = 0, binding = 1) readonly buffer ExpertWeights {
    float expert_weights[];
};

// Router output (batch, num_experts)
layout(set = 0, binding = 2) buffer RouterOutput {
    float routing_weights[];
};

// Selected experts (batch, top_k)
layout(set = 0, binding = 3) buffer SelectedExperts {
    uint expert_indices[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint batch_size;
    uint num_domains;
    uint num_experts;
    uint top_k;           // Number of experts to activate
    uint routing_mode;    // 0 = weighted sum, 1 = top-k selection
};

void main() {
    uint batch_idx = gl_GlobalInvocationID.x;
    
    if (batch_idx >= batch_size) {
        return;
    }
    
    // Compute expert routing weights based on domain probabilities
    for (uint e = 0; e < num_experts; e++) {
        float weight = 0.0;
        
        for (uint d = 0; d < num_domains; d++) {
            uint prob_idx = batch_idx * num_domains + d;
            uint expert_w_idx = d * num_experts + e;
            weight += probs[prob_idx] * expert_weights[expert_w_idx];
        }
        
        uint out_idx = batch_idx * num_experts + e;
        routing_weights[out_idx] = weight;
    }
    
    // If top-k mode, select top-k experts
    if (routing_mode == 1) {
        // Selection sort for top-k
        for (uint k = 0; k < top_k; k++) {
            float max_weight = -1e10;
            uint max_expert = 0;
            
            for (uint e = 0; e < num_experts; e++) {
                uint w_idx = batch_idx * num_experts + e;
                float w = routing_weights[w_idx];
                
                // Check if already selected
                bool selected = false;
                for (uint prev = 0; prev < k; prev++) {
                    uint sel_idx = batch_idx * top_k + prev;
                    if (expert_indices[sel_idx] == e) {
                        selected = true;
                        break;
                    }
                }
                
                if (!selected && w > max_weight) {
                    max_weight = w;
                    max_expert = e;
                }
            }
            
            uint sel_idx = batch_idx * top_k + k;
            expert_indices[sel_idx] = max_expert;
        }
    }
}
