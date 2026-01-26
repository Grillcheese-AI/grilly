#version 450

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Input token IDs (batch, seq_len)
layout(set = 0, binding = 0) readonly buffer InputTokens {
    uint token_ids[];
};

// Embedding table (vocab_size, embedding_dim)
layout(set = 0, binding = 1) readonly buffer EmbeddingTable {
    float embeddings[];
};

// Output embeddings (batch, seq_len, embedding_dim)
layout(set = 0, binding = 2) buffer OutputEmbeddings {
    float output_emb[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint batch_size;
    uint seq_len;
    uint vocab_size;
    uint embedding_dim;
};

void main() {
    uint gID = gl_GlobalInvocationID.x;
    uint total_tokens = batch_size * seq_len;
    
    if (gID >= total_tokens) {
        return;
    }
    
    // Get token ID
    uint token_id = token_ids[gID];
    
    // Bounds check
    if (token_id >= vocab_size) {
        // Invalid token, zero out embedding
        for (uint d = 0; d < embedding_dim; d++) {
            uint out_idx = gID * embedding_dim + d;
            output_emb[out_idx] = 0.0;
        }
        return;
    }
    
    // Lookup embedding and copy to output
    for (uint d = 0; d < embedding_dim; d++) {
        uint emb_idx = token_id * embedding_dim + d;
        uint out_idx = gID * embedding_dim + d;
        output_emb[out_idx] = embeddings[emb_idx];
    }
}
