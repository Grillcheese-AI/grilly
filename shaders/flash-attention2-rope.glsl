#version 450

// Flash Attention 2 with RoPE (Rotary Position Embeddings) Fused
// 
// This combines Flash Attention 2's memory-efficient tiled computation
// with RoPE applied on-the-fly during Q@K computation.
//
// Benefits:
// - No separate RoPE pass needed (fused)
// - Relative position information without explicit positional embeddings
// - Better extrapolation to longer sequences
// - Memory efficient O(N) instead of O(N²)

layout (local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

// Q, K, V inputs (batch, seq_len, num_heads, head_dim)
// Note: Q and K are NOT pre-rotated - RoPE is applied during attention
layout(set = 0, binding = 0) readonly buffer Queries {
    float Q[];
};

layout(set = 0, binding = 1) readonly buffer Keys {
    float K[];
};

layout(set = 0, binding = 2) readonly buffer Values {
    float V[];
};

// Attention mask (batch, seq_len) - 1.0 = keep, 0.0 = mask
layout(set = 0, binding = 3) readonly buffer AttentionMask {
    float mask[];
};

// Output (batch, seq_len, num_heads, head_dim)
layout(set = 0, binding = 4) buffer Output {
    float output_data[];
};

// Running max per query position
layout(set = 0, binding = 5) buffer RunningMax {
    float running_max[];
};

// Running sum per query position
layout(set = 0, binding = 6) buffer RunningSum {
    float running_sum[];
};

// Output accumulator
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
    uint tile_size_q;
    uint tile_size_k;
    uint pass_type;        // 0=init, 1=process_tile, 2=finalize
    uint has_mask;
    uint q_tile_idx;
    uint k_tile_idx;
    float rope_base;       // RoPE base (10000.0)
    float rope_scaling;    // RoPE scaling factor (1.0)
    uint use_rope;         // 1 = apply RoPE, 0 = no RoPE
};

// RoPE helper: compute theta for a dimension pair at a position
float compute_rope_theta(uint pos, uint dim_pair, float base, uint dim) {
    float freq_exp = -2.0 * float(dim_pair) / float(dim);
    return float(pos) * pow(base, freq_exp);
}

// RoPE helper: apply rotation to a pair of values
// Returns rotated values via out parameters
void apply_rope_rotation(
    float x0, float x1,           // Input pair
    float cos_theta, float sin_theta,  // Precomputed cos/sin
    out float y0, out float y1    // Output pair
) {
    y0 = x0 * cos_theta - x1 * sin_theta;
    y1 = x0 * sin_theta + x1 * cos_theta;
}

// Compute Q @ K^T with RoPE applied on-the-fly
// This computes: RoPE(Q[q_pos]) @ RoPE(K[k_pos])^T
float compute_attention_score_with_rope(
    uint batch_idx,
    uint q_pos,
    uint k_pos,
    uint head_idx
) {
    float score = 0.0;
    
    // Process dimension pairs
    for (uint d = 0; d < head_dim; d += 2) {
        uint dim_pair = d / 2;
        
        // Q indices
        uint q_idx_0 = batch_idx * seq_len * num_heads * head_dim +
                       q_pos * num_heads * head_dim +
                       head_idx * head_dim + d;
        uint q_idx_1 = q_idx_0 + 1;
        
        // K indices  
        uint k_idx_0 = batch_idx * seq_len * num_heads * head_dim +
                       k_pos * num_heads * head_dim +
                       head_idx * head_dim + d;
        uint k_idx_1 = k_idx_0 + 1;
        
        // Get Q and K values
        float q0 = Q[q_idx_0];
        float q1 = Q[q_idx_1];
        float k0 = K[k_idx_0];
        float k1 = K[k_idx_1];
        
        if (use_rope == 1) {
            // Compute theta for Q (at q_pos) and K (at k_pos)
            float theta_q = compute_rope_theta(q_pos, dim_pair, rope_base, head_dim) / rope_scaling;
            float theta_k = compute_rope_theta(k_pos, dim_pair, rope_base, head_dim) / rope_scaling;
            
            float cos_q = cos(theta_q);
            float sin_q = sin(theta_q);
            float cos_k = cos(theta_k);
            float sin_k = sin(theta_k);
            
            // Apply RoPE to Q
            float q0_rot, q1_rot;
            apply_rope_rotation(q0, q1, cos_q, sin_q, q0_rot, q1_rot);
            
            // Apply RoPE to K
            float k0_rot, k1_rot;
            apply_rope_rotation(k0, k1, cos_k, sin_k, k0_rot, k1_rot);
            
            // Dot product of rotated vectors
            score += q0_rot * k0_rot + q1_rot * k1_rot;
        } else {
            // No RoPE - standard dot product
            score += q0 * k0 + q1 * k1;
        }
    }
    
    return score * scale;
}

// Online softmax update
void update_online_softmax(
    float new_score,
    inout float max_val,
    inout float sum_val
) {
    if (new_score > max_val) {
        sum_val = sum_val * exp(max_val - new_score) + 1.0;
        max_val = new_score;
    } else {
        sum_val += exp(new_score - max_val);
    }
}

void main() {
    uint row = gl_GlobalInvocationID.y;
    uint col = gl_GlobalInvocationID.x;
    
    if (pass_type == 0) {
        // Initialize
        uint total_q = batch_size * seq_len * num_heads;
        if (row >= total_q) return;
        
        running_max[row] = -1e10;
        running_sum[row] = 0.0;
        
        for (uint d = 0; d < head_dim; d++) {
            output_accum[row * head_dim + d] = 0.0;
        }
        
    } else if (pass_type == 1) {
        // Process tile with fused RoPE
        uint q_in_tile = min(tile_size_q, seq_len - q_tile_idx * tile_size_q);
        uint k_in_tile = min(tile_size_k, seq_len - k_tile_idx * tile_size_k);
        
        uint q_offset = row % q_in_tile;
        uint k_offset = col % k_in_tile;
        
        uint q_pos = q_tile_idx * tile_size_q + q_offset;
        uint k_pos = k_tile_idx * tile_size_k + k_offset;
        
        if (q_pos >= seq_len || k_pos >= seq_len) return;
        
        uint flat_idx = row / q_in_tile;
        uint batch_idx = flat_idx / num_heads;
        uint head_idx = flat_idx % num_heads;
        
        // Compute score with fused RoPE
        float score = compute_attention_score_with_rope(batch_idx, q_pos, k_pos, head_idx);
        
        // Apply mask
        if (has_mask == 1) {
            uint mask_idx = batch_idx * seq_len + k_pos;
            score += (mask[mask_idx] - 1.0) * 1e9;
        }
        
        // Online softmax update
        uint q_flat = batch_idx * seq_len * num_heads + q_pos * num_heads + head_idx;
        
        float old_max = running_max[q_flat];
        float old_sum = running_sum[q_flat];
        float new_max = old_max;
        float new_sum = old_sum;
        
        update_online_softmax(score, new_max, new_sum);
        
        running_max[q_flat] = new_max;
        running_sum[q_flat] = new_sum;
        
        float weight = exp(score - new_max) / max(new_sum, 1e-10);
        
        // Rescale accumulator if max changed
        if (new_max > old_max) {
            float rescale = exp(old_max - new_max);
            for (uint d = 0; d < head_dim; d++) {
                output_accum[q_flat * head_dim + d] *= rescale;
            }
        }
        
        // Accumulate weighted V (V doesn't get RoPE)
        for (uint d = 0; d < head_dim; d++) {
            uint v_idx = batch_idx * seq_len * num_heads * head_dim +
                        k_pos * num_heads * head_dim +
                        head_idx * head_dim + d;
            output_accum[q_flat * head_dim + d] += weight * V[v_idx];
        }
        
    } else if (pass_type == 2) {
        // Finalize
        uint total_q = batch_size * seq_len * num_heads;
        if (row >= total_q || col >= head_dim) return;
        
        float final_sum = running_sum[row];
        float normalized = output_accum[row * head_dim + col] / max(final_sum, 1e-10);
        output_data[row * head_dim + col] = normalized;
    }
}
