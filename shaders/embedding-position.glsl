#version 450

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Token embeddings (batch, seq_len, hidden_dim)
layout(set = 0, binding = 0) readonly buffer TokenEmbeddings {
    float token_emb[];
};

// Positional embeddings (max_seq_len, hidden_dim)
layout(set = 0, binding = 1) readonly buffer PositionalEmbeddings {
    float pos_emb[];
};

// Output: token + position (batch, seq_len, hidden_dim)
layout(set = 0, binding = 2) buffer Output {
    float output_data[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint batch_size;
    uint seq_len;
    uint hidden_dim;
    uint pos_type;    // 0 = learned, 1 = sinusoidal
    float scale;      // Scaling factor for embeddings (e.g., sqrt(hidden_dim))
};

// Sinusoidal position encoding
float sinusoidal_pos(uint pos, uint dim, uint hidden_dim) {
    float position = float(pos);
    float i = float(dim / 2);
    float freq = 1.0 / pow(10000.0, 2.0 * i / float(hidden_dim));
    
    if (dim % 2 == 0) {
        return sin(position * freq);
    } else {
        return cos(position * freq);
    }
}

void main() {
    uint gID = gl_GlobalInvocationID.x;
    
    uint total_elements = batch_size * seq_len * hidden_dim;
    if (gID >= total_elements) {
        return;
    }
    
    uint batch_idx = gID / (seq_len * hidden_dim);
    uint remainder = gID % (seq_len * hidden_dim);
    uint seq_idx = remainder / hidden_dim;
    uint dim_idx = remainder % hidden_dim;
    
    // Get token embedding (scaled)
    float token_val = token_emb[gID] * scale;
    
    // Get positional embedding
    float pos_val;
    
    if (pos_type == 1) {
        // Sinusoidal (computed on-the-fly)
        pos_val = sinusoidal_pos(seq_idx, dim_idx, hidden_dim);
    } else {
        // Learned (from buffer)
        uint pos_idx = seq_idx * hidden_dim + dim_idx;
        pos_val = pos_emb[pos_idx];
    }
    
    // Add position to token embedding
    output_data[gID] = token_val + pos_val;
}
