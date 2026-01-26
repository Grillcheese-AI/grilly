#version 450

layout (local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

// Input hidden states (batch, seq_len, hidden_dim)
layout(set = 0, binding = 0) readonly buffer Input {
    float input_data[];
};

// QKV projection weights (3 * hidden_dim, hidden_dim) - combined Q, K, V
layout(set = 0, binding = 1) readonly buffer QKVWeights {
    float W_qkv[];
};

// QKV bias (3 * hidden_dim)
layout(set = 0, binding = 2) readonly buffer QKVBias {
    float b_qkv[];
};

// Output projection weights (hidden_dim, hidden_dim)
layout(set = 0, binding = 3) readonly buffer OutWeights {
    float W_out[];
};

// Output projection bias (hidden_dim)
layout(set = 0, binding = 4) readonly buffer OutBias {
    float b_out[];
};

// Attention mask (batch, seq_len) - 1.0 for valid, 0.0 for padding
layout(set = 0, binding = 5) readonly buffer Mask {
    float mask[];
};

// Scratch space for QKV (batch, seq_len, 3 * hidden_dim)
layout(set = 0, binding = 6) buffer QKVBuffer {
    float qkv[];
};

// Scratch space for attention scores (batch * num_heads, seq_len, seq_len)
layout(set = 0, binding = 7) buffer ScoresBuffer {
    float scores[];
};

// Output (batch, seq_len, hidden_dim)
layout(set = 0, binding = 8) buffer Output {
    float output_data[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint batch_size;
    uint seq_len;
    uint hidden_dim;
    uint num_heads;
    uint head_dim;      // hidden_dim / num_heads
    float scale;        // 1.0 / sqrt(head_dim)
    uint pass_type;     // 0=QKV projection, 1=attention scores, 2=softmax, 3=attention output, 4=output projection
};

// Numerically stable softmax helper
shared float shared_max[256];
shared float shared_sum[256];

void main() {
    uint gID_x = gl_GlobalInvocationID.x;
    uint gID_y = gl_GlobalInvocationID.y;
    uint lID = gl_LocalInvocationIndex;
    
    if (pass_type == 0) {
        // Pass 0: QKV projection
        // Each thread computes one element of QKV output
        uint row = gID_y;  // batch * seq
        uint col = gID_x;  // 3 * hidden_dim
        
        uint batch_seq = batch_size * seq_len;
        uint qkv_dim = 3 * hidden_dim;
        
        if (row >= batch_seq || col >= qkv_dim) return;
        
        float sum = b_qkv[col];
        for (uint k = 0; k < hidden_dim; k++) {
            uint in_idx = row * hidden_dim + k;
            uint w_idx = col * hidden_dim + k;
            sum += input_data[in_idx] * W_qkv[w_idx];
        }
        
        uint out_idx = row * qkv_dim + col;
        qkv[out_idx] = sum;
        
    } else if (pass_type == 1) {
        // Pass 1: Compute attention scores (Q @ K^T)
        // Row = batch * num_heads * seq (query position)
        // Col = key position
        uint total_query_pos = batch_size * num_heads * seq_len;
        
        if (gID_y >= total_query_pos || gID_x >= seq_len) return;
        
        uint batch_idx = gID_y / (num_heads * seq_len);
        uint remainder = gID_y % (num_heads * seq_len);
        uint head_idx = remainder / seq_len;
        uint q_pos = remainder % seq_len;
        uint k_pos = gID_x;
        
        float score = 0.0;
        
        // Q starts at offset 0, K starts at offset hidden_dim
        for (uint d = 0; d < head_dim; d++) {
            uint q_idx = batch_idx * seq_len * 3 * hidden_dim + 
                        q_pos * 3 * hidden_dim + 
                        head_idx * head_dim + d;
            
            uint k_idx = batch_idx * seq_len * 3 * hidden_dim + 
                        k_pos * 3 * hidden_dim + 
                        hidden_dim +  // K offset
                        head_idx * head_dim + d;
            
            score += qkv[q_idx] * qkv[k_idx];
        }
        
        score *= scale;
        
        // Apply mask (-inf for padding)
        uint mask_idx = batch_idx * seq_len + k_pos;
        if (mask[mask_idx] < 0.5) {
            score = -1e9;  // Large negative for softmax
        }
        
        // Store score
        uint score_idx = gID_y * seq_len + gID_x;
        scores[score_idx] = score;
        
    } else if (pass_type == 2) {
        // Pass 2: Softmax over attention scores
        // Each workgroup handles one row (query position)
        uint row = gID_y;  // batch * num_heads * seq
        uint total_rows = batch_size * num_heads * seq_len;
        
        if (row >= total_rows) return;
        
        uint base_idx = row * seq_len;
        
        // Find max (for numerical stability)
        float local_max = -1e30;
        for (uint i = lID; i < seq_len; i += gl_WorkGroupSize.x) {
            local_max = max(local_max, scores[base_idx + i]);
        }
        shared_max[lID] = local_max;
        barrier();
        
        // Reduce max
        for (uint stride = gl_WorkGroupSize.x / 2; stride > 0; stride >>= 1) {
            if (lID < stride) {
                shared_max[lID] = max(shared_max[lID], shared_max[lID + stride]);
            }
            barrier();
        }
        float row_max = shared_max[0];
        barrier();
        
        // Compute exp and sum
        float local_sum = 0.0;
        for (uint i = lID; i < seq_len; i += gl_WorkGroupSize.x) {
            float exp_val = exp(scores[base_idx + i] - row_max);
            scores[base_idx + i] = exp_val;
            local_sum += exp_val;
        }
        shared_sum[lID] = local_sum;
        barrier();
        
        // Reduce sum
        for (uint stride = gl_WorkGroupSize.x / 2; stride > 0; stride >>= 1) {
            if (lID < stride) {
                shared_sum[lID] += shared_sum[lID + stride];
            }
            barrier();
        }
        float row_sum = shared_sum[0] + 1e-9;
        barrier();
        
        // Normalize
        for (uint i = lID; i < seq_len; i += gl_WorkGroupSize.x) {
            scores[base_idx + i] /= row_sum;
        }
        
    } else if (pass_type == 3) {
        // Pass 3: Attention output (scores @ V)
        // Each thread computes one element of attended output
        uint row = gID_y;  // batch * seq
        uint col = gID_x;  // hidden_dim
        
        uint batch_seq = batch_size * seq_len;
        if (row >= batch_seq || col >= hidden_dim) return;
        
        uint batch_idx = row / seq_len;
        uint q_pos = row % seq_len;
        uint head_idx = col / head_dim;
        uint d = col % head_dim;
        
        float sum = 0.0;
        
        for (uint k_pos = 0; k_pos < seq_len; k_pos++) {
            // Get attention weight
            uint score_idx = batch_idx * num_heads * seq_len * seq_len +
                            head_idx * seq_len * seq_len +
                            q_pos * seq_len + k_pos;
            float attn = scores[score_idx];
            
            // Get V value (V offset = 2 * hidden_dim)
            uint v_idx = batch_idx * seq_len * 3 * hidden_dim +
                        k_pos * 3 * hidden_dim +
                        2 * hidden_dim +
                        head_idx * head_dim + d;
            
            sum += attn * qkv[v_idx];
        }
        
        // Store in qkv buffer temporarily (reuse space)
        uint out_idx = row * hidden_dim + col;
        qkv[out_idx] = sum;
        
    } else if (pass_type == 4) {
        // Pass 4: Output projection
        uint row = gID_y;  // batch * seq
        uint col = gID_x;  // hidden_dim
        
        uint batch_seq = batch_size * seq_len;
        if (row >= batch_seq || col >= hidden_dim) return;
        
        float sum = b_out[col];
        for (uint k = 0; k < hidden_dim; k++) {
            uint in_idx = row * hidden_dim + k;
            uint w_idx = col * hidden_dim + k;
            sum += qkv[in_idx] * W_out[w_idx];
        }
        
        // Residual connection
        uint out_idx = row * hidden_dim + col;
        output_data[out_idx] = sum + input_data[out_idx];
    }
}
