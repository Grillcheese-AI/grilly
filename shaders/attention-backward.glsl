#version 450

layout (local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

// Gradient w.r.t. attention output
layout(set = 0, binding = 0) readonly buffer GradOutput {
    float grad_output[];
};

// Query tensor (batch, heads, seq, head_dim)
layout(set = 0, binding = 1) readonly buffer Query {
    float query[];
};

// Key tensor (batch, heads, seq, head_dim)
layout(set = 0, binding = 2) readonly buffer Key {
    float key[];
};

// Value tensor (batch, heads, seq, head_dim)
layout(set = 0, binding = 3) readonly buffer Value {
    float value[];
};

// Attention weights (softmax output) from forward pass
layout(set = 0, binding = 4) readonly buffer AttnWeights {
    float attn_weights[];
};

// Output: gradient w.r.t. query
layout(set = 0, binding = 5) buffer GradQuery {
    float grad_query[];
};

// Output: gradient w.r.t. key
layout(set = 0, binding = 6) buffer GradKey {
    float grad_key[];
};

// Output: gradient w.r.t. value
layout(set = 0, binding = 7) buffer GradValue {
    float grad_value[];
};

// Scratch buffer for intermediate gradients
layout(set = 0, binding = 8) buffer Scratch {
    float scratch[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint batch_size;
    uint num_heads;
    uint seq_len;
    uint head_dim;
    float scale;      // 1.0 / sqrt(head_dim)
    uint pass_type;   // 0 = grad_V, 1 = grad_attn, 2 = grad_Q/K
};

// Indexing helpers
uint qkv_idx(uint b, uint h, uint s, uint d) {
    return b * num_heads * seq_len * head_dim +
           h * seq_len * head_dim +
           s * head_dim + d;
}

uint attn_idx(uint b, uint h, uint i, uint j) {
    return b * num_heads * seq_len * seq_len +
           h * seq_len * seq_len +
           i * seq_len + j;
}

void main() {
    uint gx = gl_GlobalInvocationID.x;
    uint gy = gl_GlobalInvocationID.y;

    if (pass_type == 0) {
        // Pass 1: Compute gradient w.r.t. Value
        // grad_V = attn_weights^T @ grad_output
        // For each (batch, head, seq_j, dim_d): grad_V[b,h,j,d] = sum_i attn[b,h,i,j] * grad_out[b,h,i,d]

        uint total_positions = batch_size * num_heads * seq_len;
        if (gx >= total_positions || gy >= head_dim) return;

        uint b = gx / (num_heads * seq_len);
        uint remainder = gx % (num_heads * seq_len);
        uint h = remainder / seq_len;
        uint j = remainder % seq_len;  // This is the key/value position
        uint d = gy;

        float sum = 0.0;
        for (uint i = 0; i < seq_len; i++) {
            // attn_weights[b,h,i,j] * grad_output[b,h,i,d]
            uint a_idx = attn_idx(b, h, i, j);
            uint go_idx = qkv_idx(b, h, i, d);
            sum += attn_weights[a_idx] * grad_output[go_idx];
        }

        uint gv_idx = qkv_idx(b, h, j, d);
        grad_value[gv_idx] = sum;

    } else if (pass_type == 1) {
        // Pass 2: Compute gradient w.r.t. attention weights (pre-softmax scores)
        // First: grad_attn_weights = grad_output @ V^T
        // Then: grad_scores = softmax_backward(grad_attn_weights, attn_weights)

        uint total_positions = batch_size * num_heads * seq_len;
        if (gx >= total_positions || gy >= seq_len) return;

        uint b = gx / (num_heads * seq_len);
        uint remainder = gx % (num_heads * seq_len);
        uint h = remainder / seq_len;
        uint i = remainder % seq_len;  // Query position
        uint j = gy;                    // Key position

        // Compute grad_attn[i,j] = sum_d grad_output[i,d] * V[j,d]
        float grad_attn_raw = 0.0;
        for (uint d = 0; d < head_dim; d++) {
            uint go_idx = qkv_idx(b, h, i, d);
            uint v_idx = qkv_idx(b, h, j, d);
            grad_attn_raw += grad_output[go_idx] * value[v_idx];
        }

        // Apply softmax backward: grad_score = attn * (grad_attn - sum_k(attn * grad_attn))
        // First compute the dot product for this row
        uint a_idx = attn_idx(b, h, i, j);
        float attn_val = attn_weights[a_idx];

        // Store raw grad_attn in scratch for reduction
        scratch[a_idx] = grad_attn_raw;

    } else if (pass_type == 2) {
        // Pass 3: Complete softmax backward and compute grad_Q, grad_K

        uint total_positions = batch_size * num_heads * seq_len;
        if (gx >= total_positions || gy >= head_dim) return;

        uint b = gx / (num_heads * seq_len);
        uint remainder = gx % (num_heads * seq_len);
        uint h = remainder / seq_len;
        uint i = remainder % seq_len;  // Position index
        uint d = gy;                    // Dimension index

        // Compute softmax backward for row i
        // dot_i = sum_j attn[i,j] * grad_attn_raw[i,j]
        float dot_i = 0.0;
        for (uint j = 0; j < seq_len; j++) {
            uint a_idx = attn_idx(b, h, i, j);
            dot_i += attn_weights[a_idx] * scratch[a_idx];
        }

        // grad_Q[i,d] = scale * sum_j grad_scores[i,j] * K[j,d]
        // grad_K[j,d] = scale * sum_i grad_scores[i,j] * Q[i,d]

        float grad_q_val = 0.0;
        float grad_k_val = 0.0;

        for (uint j = 0; j < seq_len; j++) {
            uint a_idx_ij = attn_idx(b, h, i, j);
            float attn_ij = attn_weights[a_idx_ij];
            float grad_attn_ij = scratch[a_idx_ij];

            // grad_scores[i,j] = attn[i,j] * (grad_attn[i,j] - dot_i)
            float grad_score_ij = attn_ij * (grad_attn_ij - dot_i) * scale;

            // Accumulate grad_Q
            uint k_jd = qkv_idx(b, h, j, d);
            grad_q_val += grad_score_ij * key[k_jd];
        }

        // For grad_K, we need grad_scores[i,j] for fixed j and varying i
        // This is computed differently - each thread computes grad_K[i,d]
        for (uint query_i = 0; query_i < seq_len; query_i++) {
            uint a_idx_qi = attn_idx(b, h, query_i, i);  // query_i attends to position i
            float attn_qi = attn_weights[a_idx_qi];
            float grad_attn_qi = scratch[a_idx_qi];

            // Recompute dot for query_i
            float dot_qi = 0.0;
            for (uint k = 0; k < seq_len; k++) {
                uint a_idx_qk = attn_idx(b, h, query_i, k);
                dot_qi += attn_weights[a_idx_qk] * scratch[a_idx_qk];
            }

            float grad_score_qi = attn_qi * (grad_attn_qi - dot_qi) * scale;

            uint q_id = qkv_idx(b, h, query_i, d);
            grad_k_val += grad_score_qi * query[q_id];
        }

        uint idx = qkv_idx(b, h, i, d);
        grad_query[idx] = grad_q_val;
        grad_key[idx] = grad_k_val;
    }
}
