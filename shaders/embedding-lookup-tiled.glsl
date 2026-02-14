#version 450

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

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

layout(push_constant) uniform PushConsts {
    uint batch_size;
    uint seq_len;
    uint vocab_size;
    uint embedding_dim;
};

void main() {
    uint token_idx = gl_GlobalInvocationID.x;
    uint dim_idx = gl_GlobalInvocationID.y;
    uint total_tokens = batch_size * seq_len;

    if (token_idx >= total_tokens || dim_idx >= embedding_dim) {
        return;
    }

    uint token_id = token_ids[token_idx];
    uint out_idx = token_idx * embedding_dim + dim_idx;

    if (token_id >= vocab_size) {
        output_emb[out_idx] = 0.0;
        return;
    }

    uint emb_idx = token_id * embedding_dim + dim_idx;
    output_emb[out_idx] = embeddings[emb_idx];
}
