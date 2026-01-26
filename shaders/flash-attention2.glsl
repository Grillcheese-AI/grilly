#version 450

// Flash Attention 2: Tiled attention with online softmax
// Processes attention in blocks to reduce memory usage from O(N²) to O(N)
// Uses online softmax algorithm for numerical stability

layout (local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

// Query, Key, Value inputs (batch, seq_len, num_heads, head_dim)
layout(set = 0, binding = 0) readonly buffer Queries {
    float Q[];
};

layout(set = 0, binding = 1) readonly buffer Keys {
    float K[];
};

layout(set = 0, binding = 2) readonly buffer Values {
    float V[];
};

// Optional attention mask (batch, seq_len) - 0.0 = mask out, 1.0 = keep
layout(set = 0, binding = 3) readonly buffer AttentionMask {
    float mask[];
};

// Output (batch, seq_len, num_heads, head_dim)
layout(set = 0, binding = 4) buffer Output {
    float output_data[];
};

// Temporary buffers for online softmax (per query position)
// Running max for each query position
layout(set = 0, binding = 5) buffer RunningMax {
    float running_max[];
};

// Running sum of exp(scores - max) for each query position
layout(set = 0, binding = 6) buffer RunningSum {
    float running_sum[];
};

// Temporary output accumulator (batch, seq_len, num_heads, head_dim)
layout(set = 0, binding = 7) buffer OutputAccumulator {
    float output_accum[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint batch_size;
    uint seq_len;
    uint num_heads;
    uint head_dim;
    float scale;           // 1 / sqrt(head_dim)
    uint tile_size_q;       // Tile size for query dimension (typically 64-128)
    uint tile_size_k;       // Tile size for key dimension (typically 64-128)
    uint pass_type;         // 0 = initialize, 1 = process tile, 2 = finalize
    uint has_mask;          // 1 if mask is provided, 0 otherwise
    uint q_tile_idx;        // Current query tile index (for pass_type=1)
    uint k_tile_idx;        // Current key tile index (for pass_type=1)
};

// Online softmax update: updates running max and sum incrementally
// Based on: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
void update_online_softmax(
    float new_score,
    inout float running_max_val,
    inout float running_sum_val
) {
    if (new_score > running_max_val) {
        // New max found: rescale existing sum
        float exp_old = exp(running_max_val - new_score);
        running_sum_val = running_sum_val * exp_old + exp(0.0); // exp(new_score - new_score) = 1
        running_max_val = new_score;
    } else {
        // Add new score to existing sum
        running_sum_val += exp(new_score - running_max_val);
    }
}

void main() {
    uint row = gl_GlobalInvocationID.y;
    uint col = gl_GlobalInvocationID.x;
    
    if (pass_type == 0) {
        // Pass 0: Initialize running max and sum for each query position
        uint total_q_positions = batch_size * seq_len * num_heads;
        
        if (row >= total_q_positions) {
            return;
        }
        
        // Initialize running max to very negative value
        uint max_idx = row;
        running_max[max_idx] = -1e10;
        
        // Initialize running sum to 0
        running_sum[max_idx] = 0.0;
        
        // Initialize output accumulator to zeros
        for (uint d = 0; d < head_dim; d++) {
            uint out_idx = row * head_dim + d;
            output_accum[out_idx] = 0.0;
        }
        
    } else if (pass_type == 1) {
        // Pass 1: Process a tile of attention
        // Each thread processes one (query_pos, key_pos) pair within the current tile
        
        uint total_q_in_tile = min(tile_size_q, seq_len - q_tile_idx * tile_size_q);
        uint total_k_in_tile = min(tile_size_k, seq_len - k_tile_idx * tile_size_k);
        
        // Decode thread position within tile
        uint q_offset_in_tile = row % total_q_in_tile;
        uint k_offset_in_tile = col % total_k_in_tile;
        
        // Global positions
        uint q_pos = q_tile_idx * tile_size_q + q_offset_in_tile;
        uint k_pos = k_tile_idx * tile_size_k + k_offset_in_tile;
        
        if (q_pos >= seq_len || k_pos >= seq_len) {
            return;
        }
        
        // Decode batch and head from row
        // Row represents: (batch * num_heads * total_q_in_tile) + (head * total_q_in_tile) + q_offset_in_tile
        uint flat_idx = row / total_q_in_tile;
        uint batch_idx = flat_idx / num_heads;
        uint head_idx = flat_idx % num_heads;
        
        // Compute attention score: Q[q_pos] @ K[k_pos]^T / sqrt(head_dim)
        float score = 0.0;
        
        for (uint d = 0; d < head_dim; d++) {
            // Q index: [batch, seq, head, head_dim]
            uint q_idx = batch_idx * seq_len * num_heads * head_dim +
                        q_pos * num_heads * head_dim +
                        head_idx * head_dim + d;
            
            // K index: [batch, seq, head, head_dim]
            uint k_idx = batch_idx * seq_len * num_heads * head_dim +
                        k_pos * num_heads * head_dim +
                        head_idx * head_dim + d;
            
            score += Q[q_idx] * K[k_idx];
        }
        
        score *= scale;
        
        // Apply mask if provided
        if (has_mask == 1) {
            uint mask_idx = batch_idx * seq_len + k_pos;
            score += (mask[mask_idx] - 1.0) * 1e9; // Large negative if masked
        }
        
        // Get running max and sum for this query position
        uint q_flat_idx = batch_idx * seq_len * num_heads +
                         q_pos * num_heads +
                         head_idx;
        
        float old_max = running_max[q_flat_idx];
        float old_sum = running_sum[q_flat_idx];
        
        // Update online softmax
        float new_max = old_max;
        float new_sum = old_sum;
        update_online_softmax(score, new_max, new_sum);
        
        // Store updated max and sum
        running_max[q_flat_idx] = new_max;
        running_sum[q_flat_idx] = new_sum;
        
        // Compute attention weight: exp(score - max) / sum
        float weight = exp(score - new_max) / new_sum;
        
        // Rescale existing accumulator if max changed
        if (new_max > old_max) {
            float rescale = exp(old_max - new_max);
            for (uint d = 0; d < head_dim; d++) {
                uint accum_idx = q_flat_idx * head_dim + d;
                output_accum[accum_idx] *= rescale;
            }
        }
        
        // Accumulate weighted value: output += weight * V[k_pos]
        for (uint d = 0; d < head_dim; d++) {
            uint accum_idx = q_flat_idx * head_dim + d;
            
            // V index: [batch, seq, head, head_dim]
            uint v_idx = batch_idx * seq_len * num_heads * head_dim +
                        k_pos * num_heads * head_dim +
                        head_idx * head_dim + d;
            
            // Accumulate: output[q_pos] += weight * V[k_pos]
            output_accum[accum_idx] += weight * V[v_idx];
        }
        
    } else if (pass_type == 2) {
        // Pass 2: Finalize output (normalize by final sum and copy to output)
        uint total_q_positions = batch_size * seq_len * num_heads;
        
        if (row >= total_q_positions || col >= head_dim) {
            return;
        }
        
        uint q_flat_idx = row;
        uint d = col;
        
        // Get final sum
        float final_sum = running_sum[q_flat_idx];
        
        // Normalize accumulator by final sum (if sum changed during processing)
        uint accum_idx = q_flat_idx * head_dim + d;
        float normalized = output_accum[accum_idx] / max(final_sum, 1e-10);
        
        // Copy to output
        output_data[accum_idx] = normalized;
    }
}
