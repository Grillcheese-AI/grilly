#version 450

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Query vectors (batch, query_dim)
layout(set = 0, binding = 0) readonly buffer Queries {
    float queries[];
};

// Memory keys (num_memories, key_dim)
layout(set = 0, binding = 1) readonly buffer MemoryKeys {
    float keys[];
};

// Memory values (num_memories, value_dim)
layout(set = 0, binding = 2) readonly buffer MemoryValues {
    float values[];
};

// Output retrieved values (batch, value_dim)
layout(set = 0, binding = 3) buffer Output {
    float output_data[];
};

// Attention scores (batch, num_memories) - temporary
layout(set = 0, binding = 4) buffer AttentionScores {
    float scores[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint batch_size;
    uint num_memories;
    uint key_dim;
    uint value_dim;
    float temperature;      // Temperature for softmax (e.g., sqrt(key_dim))
    uint pass_type;         // 0 = compute scores, 1 = softmax, 2 = weighted sum
};

void main() {
    uint gID = gl_GlobalInvocationID.x;
    
    if (pass_type == 0) {
        // Pass 1: Compute attention scores (cosine similarity or dot product)
        uint total_scores = batch_size * num_memories;
        if (gID >= total_scores) return;
        
        uint batch_idx = gID / num_memories;
        uint mem_idx = gID % num_memories;
        
        // Dot product: score = query · key
        float score = 0.0;
        for (uint k = 0; k < key_dim; k++) {
            uint q_idx = batch_idx * key_dim + k;
            uint k_idx = mem_idx * key_dim + k;
            score += queries[q_idx] * keys[k_idx];
        }
        
        // Scale by temperature
        scores[gID] = score / temperature;
        
    } else if (pass_type == 2) {
        // Pass 3: Weighted sum of values
        uint total_outputs = batch_size * value_dim;
        if (gID >= total_outputs) return;
        
        uint batch_idx = gID / value_dim;
        uint val_idx = gID % value_dim;
        
        float weighted_sum = 0.0;
        for (uint m = 0; m < num_memories; m++) {
            uint score_idx = batch_idx * num_memories + m;
            uint value_idx = m * value_dim + val_idx;
            weighted_sum += scores[score_idx] * values[value_idx];
        }
        
        output_data[gID] = weighted_sum;
    }
}
