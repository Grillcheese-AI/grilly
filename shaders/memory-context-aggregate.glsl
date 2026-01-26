#version 450

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Memory features (batch, num_retrieved, dim)
layout(set = 0, binding = 0) readonly buffer MemoryFeatures {
    float memory_feats[];
};

// Memory scores (batch, num_retrieved)
layout(set = 0, binding = 1) readonly buffer MemoryScores {
    float scores[];
};

// Softmax scores (batch, num_retrieved) - output
layout(set = 0, binding = 2) buffer SoftmaxScores {
    float softmax_scores[];
};

// Weighted memory context (batch, dim)
layout(set = 0, binding = 3) buffer MemoryContext {
    float context[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint batch_size;
    uint num_retrieved;
    uint dim;
    uint pass_type;      // 0 = softmax, 1 = weighted sum
};

void main() {
    uint gID = gl_GlobalInvocationID.x;
    
    if (pass_type == 0) {
        // Pass 1: Compute softmax over scores
        if (gID >= batch_size) return;
        
        uint batch_idx = gID;
        
        // Find max for numerical stability
        float max_score = -1e10;
        for (uint k = 0; k < num_retrieved; k++) {
            uint score_idx = batch_idx * num_retrieved + k;
            max_score = max(max_score, scores[score_idx]);
        }
        
        // Compute exp and sum
        float sum_exp = 0.0;
        for (uint k = 0; k < num_retrieved; k++) {
            uint score_idx = batch_idx * num_retrieved + k;
            float exp_val = exp(scores[score_idx] - max_score);
            softmax_scores[score_idx] = exp_val;
            sum_exp += exp_val;
        }
        
        // Normalize
        for (uint k = 0; k < num_retrieved; k++) {
            uint score_idx = batch_idx * num_retrieved + k;
            softmax_scores[score_idx] /= sum_exp;
        }
        
    } else if (pass_type == 1) {
        // Pass 2: Weighted sum of memory features
        uint total_elements = batch_size * dim;
        if (gID >= total_elements) return;
        
        uint batch_idx = gID / dim;
        uint dim_idx = gID % dim;
        
        float weighted_sum = 0.0;
        for (uint k = 0; k < num_retrieved; k++) {
            uint score_idx = batch_idx * num_retrieved + k;
            uint feat_idx = batch_idx * num_retrieved * dim + k * dim + dim_idx;
            
            weighted_sum += softmax_scores[score_idx] * memory_feats[feat_idx];
        }
        
        context[gID] = weighted_sum;
    }
}
