#version 450

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Hidden states (batch, seq_len, dim)
layout(set = 0, binding = 0) readonly buffer HiddenStates {
    float hidden[];
};

// Query projection weights (dim, dim)
layout(set = 0, binding = 1) readonly buffer QueryWeights {
    float W_query[];
};

// Query projection bias (dim)
layout(set = 0, binding = 2) readonly buffer QueryBias {
    float b_query[];
};

// Output query vectors (batch, dim) - mean pooled and projected
layout(set = 0, binding = 3) buffer OutputQueries {
    float queries[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint batch_size;
    uint seq_len;
    uint dim;
};

void main() {
    uint gID = gl_GlobalInvocationID.x;
    uint total_queries = batch_size * dim;
    
    if (gID >= total_queries) {
        return;
    }
    
    uint batch_idx = gID / dim;
    uint dim_idx = gID % dim;
    
    // Step 1: Mean pool over sequence
    float mean_val = 0.0;
    for (uint t = 0; t < seq_len; t++) {
        uint hidden_idx = batch_idx * seq_len * dim + t * dim + dim_idx;
        mean_val += hidden[hidden_idx];
    }
    mean_val /= float(seq_len);
    
    // Step 2: Linear projection: query = W @ mean_pooled + b
    float query_val = 0.0;
    for (uint k = 0; k < dim; k++) {
        uint mean_idx = batch_idx * dim + k;
        
        // Compute mean for dimension k
        float mean_k = 0.0;
        for (uint t = 0; t < seq_len; t++) {
            uint h_idx = batch_idx * seq_len * dim + t * dim + k;
            mean_k += hidden[h_idx];
        }
        mean_k /= float(seq_len);
        
        uint w_idx = dim_idx * dim + k;
        query_val += W_query[w_idx] * mean_k;
    }
    
    query_val += b_query[dim_idx];
    queries[gID] = query_val;
}
