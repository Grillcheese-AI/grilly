#version 450

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Token embeddings (batch, seq_len, embedding_dim)
layout(set = 0, binding = 0) readonly buffer TokenEmbeddings {
    float token_emb[];
};

// Place cell activity from semantic encoder (batch, seq_len, num_place_cells)
layout(set = 0, binding = 1) readonly buffer PlaceCellActivity {
    float place_activity[];
};

// Place cell projection weights (embedding_dim, num_place_cells)
layout(set = 0, binding = 2) readonly buffer PlaceProjection {
    float W_place[];
};

// Output semantic embeddings (batch, seq_len, embedding_dim)
layout(set = 0, binding = 3) buffer SemanticEmbeddings {
    float semantic_emb[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint batch_size;
    uint seq_len;
    uint embedding_dim;
    uint num_place_cells;
    float place_strength;    // Strength of place cell contribution (e.g., 0.1)
};

void main() {
    uint gID = gl_GlobalInvocationID.x;
    uint total_elements = batch_size * seq_len * embedding_dim;
    
    if (gID >= total_elements) {
        return;
    }
    
    // Decode position
    uint batch_idx = gID / (seq_len * embedding_dim);
    uint remainder = gID % (seq_len * embedding_dim);
    uint seq_idx = remainder / embedding_dim;
    uint emb_idx = remainder % embedding_dim;
    
    // Get token embedding
    float token_val = token_emb[gID];
    
    // Compute place cell contribution
    float place_contribution = 0.0;
    
    for (uint p = 0; p < num_place_cells; p++) {
        uint place_act_idx = batch_idx * seq_len * num_place_cells + 
                            seq_idx * num_place_cells + p;
        uint proj_idx = emb_idx * num_place_cells + p;
        
        place_contribution += place_activity[place_act_idx] * W_place[proj_idx];
    }
    
    // Combine: semantic = token_embedding + strength * place_cell_projection
    semantic_emb[gID] = token_val + place_strength * place_contribution;
}
