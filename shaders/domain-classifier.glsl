#version 450

layout (local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

// Input features/embeddings (batch, seq_len, dim)
layout(set = 0, binding = 0) readonly buffer InputFeatures {
    float features[];
};

// Classifier weights (num_domains, dim)
layout(set = 0, binding = 1) readonly buffer ClassifierWeights {
    float W[];
};

// Classifier bias (num_domains)
layout(set = 0, binding = 2) readonly buffer ClassifierBias {
    float b[];
};

// Domain logits output (batch, seq_len, num_domains)
layout(set = 0, binding = 3) buffer DomainLogits {
    float logits[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint batch_size;
    uint seq_len;
    uint dim;
    uint num_domains;
    uint pooling_type;   // 0 = mean, 1 = max, 2 = first token, 3 = per-token
};

void main() {
    uint row = gl_GlobalInvocationID.y;
    uint col = gl_GlobalInvocationID.x;
    
    if (pooling_type == 3) {
        // Per-token classification
        uint total_positions = batch_size * seq_len;
        if (row >= total_positions || col >= num_domains) {
            return;
        }
        
        uint batch_idx = row / seq_len;
        uint seq_idx = row % seq_len;
        
        // Compute logit: W[domain] · features[position] + b[domain]
        float logit = 0.0;
        for (uint d = 0; d < dim; d++) {
            uint feat_idx = batch_idx * seq_len * dim + seq_idx * dim + d;
            uint w_idx = col * dim + d;
            logit += features[feat_idx] * W[w_idx];
        }
        logit += b[col];
        
        uint out_idx = batch_idx * seq_len * num_domains + seq_idx * num_domains + col;
        logits[out_idx] = logit;
        
    } else {
        // Sequence-level classification (with pooling)
        if (row >= batch_size || col >= num_domains) {
            return;
        }
        
        uint batch_idx = row;
        uint domain_idx = col;
        
        // Pool features over sequence
        float pooled_feat[256];  // Assuming dim <= 256
        
        for (uint d = 0; d < dim && d < 256; d++) {
            float pool_val = 0.0;
            
            if (pooling_type == 0) {
                // Mean pooling
                for (uint t = 0; t < seq_len; t++) {
                    uint feat_idx = batch_idx * seq_len * dim + t * dim + d;
                    pool_val += features[feat_idx];
                }
                pool_val /= float(seq_len);
                
            } else if (pooling_type == 1) {
                // Max pooling
                pool_val = -1e10;
                for (uint t = 0; t < seq_len; t++) {
                    uint feat_idx = batch_idx * seq_len * dim + t * dim + d;
                    pool_val = max(pool_val, features[feat_idx]);
                }
                
            } else if (pooling_type == 2) {
                // First token (CLS-style)
                uint feat_idx = batch_idx * seq_len * dim + d;
                pool_val = features[feat_idx];
            }
            
            pooled_feat[d] = pool_val;
        }
        
        // Compute logit
        float logit = 0.0;
        for (uint d = 0; d < dim && d < 256; d++) {
            uint w_idx = domain_idx * dim + d;
            logit += pooled_feat[d] * W[w_idx];
        }
        logit += b[domain_idx];
        
        uint out_idx = batch_idx * num_domains + domain_idx;
        logits[out_idx] = logit;
    }
}
