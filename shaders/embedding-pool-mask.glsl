#version 450

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Input embeddings (batch, seq_len, dim)
layout(set = 0, binding = 0) readonly buffer Input {
    float embeddings[];
};

// Attention mask (batch, seq_len) - 1.0 = keep, 0.0 = mask out
layout(set = 0, binding = 1) readonly buffer Mask {
    float mask[];
};

// Output pooled embeddings (batch, dim)
layout(set = 0, binding = 2) buffer Output {
    float output_data[];
};

// Temporary buffer for mask sums (batch,)
layout(set = 0, binding = 3) buffer MaskSums {
    float mask_sums[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint batch_size;
    uint seq_len;
    uint dim;
    uint pool_type;  // 0 = mean, 1 = max, 2 = sum
};

shared float shared_sum[256];
shared float shared_max[256];

void main() {
    uint gID = gl_GlobalInvocationID.x;
    uint lID = gl_LocalInvocationIndex;
    
    if (pool_type == 0) {
        // Mean pooling with mask
        // Each workgroup handles one batch element and one dimension
        uint batch_idx = gID / dim;
        uint dim_idx = gID % dim;
        
        if (batch_idx >= batch_size || dim_idx >= dim) return;
        
        // First, compute mask sum for this batch element (only once per batch)
        if (dim_idx == 0 && lID == 0) {
            float mask_sum = 0.0;
            for (uint s = 0; s < seq_len; s++) {
                uint mask_idx = batch_idx * seq_len + s;
                mask_sum += mask[mask_idx];
            }
            mask_sums[batch_idx] = mask_sum;
        }
        barrier();
        
        // Compute weighted sum for this dimension
        float weighted_sum = 0.0;
        for (uint s = 0; s < seq_len; s++) {
            uint emb_idx = batch_idx * seq_len * dim + s * dim + dim_idx;
            uint mask_idx = batch_idx * seq_len + s;
            
            weighted_sum += embeddings[emb_idx] * mask[mask_idx];
        }
        
        // Divide by mask sum (mean pooling)
        float mask_sum = mask_sums[batch_idx];
        uint out_idx = batch_idx * dim + dim_idx;
        output_data[out_idx] = weighted_sum / (mask_sum + 1e-8);
        
    } else if (pool_type == 1) {
        // Max pooling with mask
        uint batch_idx = gID / dim;
        uint dim_idx = gID % dim;
        
        if (batch_idx >= batch_size || dim_idx >= dim) return;
        
        float max_val = -1e10;
        bool has_valid = false;
        
        for (uint s = 0; s < seq_len; s++) {
            uint mask_idx = batch_idx * seq_len + s;
            if (mask[mask_idx] > 0.5) {  // Only consider unmasked positions
                uint emb_idx = batch_idx * seq_len * dim + s * dim + dim_idx;
                max_val = max(max_val, embeddings[emb_idx]);
                has_valid = true;
            }
        }
        
        uint out_idx = batch_idx * dim + dim_idx;
        output_data[out_idx] = has_valid ? max_val : 0.0;
        
    } else if (pool_type == 2) {
        // Sum pooling with mask
        uint batch_idx = gID / dim;
        uint dim_idx = gID % dim;
        
        if (batch_idx >= batch_size || dim_idx >= dim) return;
        
        float sum = 0.0;
        for (uint s = 0; s < seq_len; s++) {
            uint emb_idx = batch_idx * seq_len * dim + s * dim + dim_idx;
            uint mask_idx = batch_idx * seq_len + s;
            
            sum += embeddings[emb_idx] * mask[mask_idx];
        }
        
        uint out_idx = batch_idx * dim + dim_idx;
        output_data[out_idx] = sum;
    }
}
