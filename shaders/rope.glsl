#version 450

// RoPE (Rotary Position Embeddings) Shader
// Applies rotary embeddings to Q and K before attention
// ModernBERT uses rotate_half approach:
// rotate_half(x) = [-x[head_dim//2:], x[:head_dim//2]]
// x_embed = (x * cos) + (rotate_half(x) * sin)

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Input Q or K (batch, seq_len, num_heads, head_dim)
layout(set = 0, binding = 0) readonly buffer Input {
    float input_data[];
};

// Output rotated Q or K (same shape)
layout(set = 0, binding = 1) buffer Output {
    float output_data[];
};

// Precomputed cos/sin tables (optional - can compute on-the-fly)
// Shape: (max_seq_len, head_dim/2) for cos, same for sin
layout(set = 0, binding = 2) readonly buffer CosTable {
    float cos_table[];
};

layout(set = 0, binding = 3) readonly buffer SinTable {
    float sin_table[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint batch_size;
    uint seq_len;
    uint num_heads;
    uint head_dim;
    float rope_base;       // Base for frequency computation (default: 10000.0)
    uint use_precomputed;  // 1 = use tables, 0 = compute on-the-fly
    float rope_scaling;    // For extended context (default: 1.0)
};

// Compute RoPE frequency for dimension pair i at position pos
// For ModernBERT: inv_freq = 1.0 / (rope_base ^ (arange(0, head_dim, 2) / head_dim))
// freqs = pos * inv_freq, then expand to [freqs, freqs]
float compute_theta(uint pos, uint freq_idx, float base, uint head_dim_val, float scaling) {
    float position = float(pos) / scaling;
    float freq_exp = -2.0 * float(freq_idx) / float(head_dim_val);
    float freq = pow(base, freq_exp);
    return position * freq;
}

void main() {
    uint gID = gl_GlobalInvocationID.x;
    
    // Total elements = batch * seq * heads * head_dim
    uint total_elements = batch_size * seq_len * num_heads * head_dim;
    
    if (gID >= total_elements) return;
    
    // Decode position: gID = batch_idx * (seq_len * num_heads * head_dim) + 
    //                        seq_idx * (num_heads * head_dim) + 
    //                        head_idx * head_dim + 
    //                        dim_idx
    uint batch_idx = gID / (seq_len * num_heads * head_dim);
    uint remainder = gID % (seq_len * num_heads * head_dim);
    uint seq_idx = remainder / (num_heads * head_dim);
    remainder = remainder % (num_heads * head_dim);
    uint head_idx = remainder / head_dim;
    uint dim_idx = remainder % head_dim;
    
    // Get current element
    float x_current = input_data[gID];
    
    // Compute which element in rotate_half corresponds to this position
    // rotate_half(x) = [-x[head_dim//2:], x[:head_dim//2]]
    uint half_dim = head_dim / 2;
    uint rotated_idx;
    float rotated_sign;
    
    if (dim_idx < half_dim) {
        // First half: rotate_half maps to second half (negated)
        // Index in second half = same position in second half
        rotated_idx = batch_idx * (seq_len * num_heads * head_dim) +
                      seq_idx * (num_heads * head_dim) +
                      head_idx * head_dim +
                      (half_dim + dim_idx);
        rotated_sign = -1.0;
    } else {
        // Second half: rotate_half maps to first half
        // Index in first half = dim_idx - half_dim
        rotated_idx = batch_idx * (seq_len * num_heads * head_dim) +
                      seq_idx * (num_heads * head_dim) +
                      head_idx * head_dim +
                      (dim_idx - half_dim);
        rotated_sign = 1.0;
    }
    
    float x_rotated = input_data[rotated_idx] * rotated_sign;
    
    // Compute cos/sin for this position
    // For ModernBERT: inv_freq = 1.0 / (rope_base ^ (arange(0, head_dim, 2) / head_dim))
    // freqs = pos * inv_freq, then expand to [freqs, freqs]
    // cos/sin are computed from freqs
    
    // We need to compute which frequency pair this dimension belongs to
    uint freq_idx = (dim_idx < half_dim) ? dim_idx : (dim_idx - half_dim);
    
    float cos_val, sin_val;
    if (use_precomputed == 1) {
        // Use precomputed tables
        uint table_idx = seq_idx * (head_dim / 2) + freq_idx;
        cos_val = cos_table[table_idx];
        sin_val = sin_table[table_idx];
    } else {
        // Compute on-the-fly
        float position = float(seq_idx) / rope_scaling;
        float freq_exp = -2.0 * float(freq_idx) / float(head_dim);
        float freq = pow(rope_base, freq_exp);
        float theta = position * freq;
        cos_val = cos(theta);
        sin_val = sin(theta);
    }
    
    // Apply: x_embed = (x * cos) + (rotate_half(x) * sin)
    output_data[gID] = x_current * cos_val + x_rotated * sin_val;
}
