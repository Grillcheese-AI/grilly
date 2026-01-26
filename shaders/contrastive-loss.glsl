#version 450

// Contrastive Loss Computation Shader
// Computes triplet contrastive loss: max(0, pos_dist - neg_dist + margin)
// 
// For each anchor:
// - Compute cosine similarity to positive
// - Compute cosine similarity to all negatives
// - Find hardest negative (highest similarity)
// - Compute triplet loss

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Anchor embeddings (batch, dim)
layout(set = 0, binding = 0) readonly buffer Anchors {
    float anchors[];
};

// Positive embeddings (batch, dim)
layout(set = 0, binding = 1) readonly buffer Positives {
    float positives[];
};

// Negative embeddings (num_negatives, dim)
layout(set = 0, binding = 2) readonly buffer Negatives {
    float negatives[];
};

// Output losses (batch,)
layout(set = 0, binding = 3) buffer Losses {
    float losses[];
};

// Output: hardest negative index per anchor (batch,)
layout(set = 0, binding = 4) buffer HardestIndices {
    int hardest_idx[];
};

// Output: positive distances (batch,)
layout(set = 0, binding = 5) buffer PosDists {
    float pos_dists[];
};

// Output: hardest negative distances (batch,)
layout(set = 0, binding = 6) buffer NegDists {
    float neg_dists[];
};

layout(push_constant) uniform PushConsts {
    uint batch_size;
    uint embedding_dim;
    uint num_negatives;
    float margin;
};

shared float shared_data[256];

// Compute cosine similarity between two vectors
float cosine_similarity(uint idx1, uint idx2, uint dim, bool is_negative) {
    float dot_product = 0.0;
    float norm1_sq = 0.0;
    float norm2_sq = 0.0;
    
    for (uint d = 0; d < dim; d++) {
        float a = anchors[idx1 * dim + d];
        float b = is_negative ? negatives[idx2 * dim + d] : positives[idx2 * dim + d];
        dot_product += a * b;
        norm1_sq += a * a;
        norm2_sq += b * b;
    }
    
    float norm = sqrt(norm1_sq * norm2_sq + 1e-8);
    return dot_product / norm;
}

void main() {
    uint batch_idx = gl_GlobalInvocationID.x;
    
    if (batch_idx >= batch_size) return;
    
    // Compute positive similarity (higher = more similar)
    float pos_sim = cosine_similarity(batch_idx, batch_idx, embedding_dim, false);
    float pos_dist = 1.0 - pos_sim;  // Distance = 1 - similarity
    
    // Find hardest negative (highest similarity = smallest distance)
    float max_neg_sim = -1.0;
    int hardest = 0;
    
    for (uint n = 0; n < num_negatives; n++) {
        float neg_sim = cosine_similarity(batch_idx, n, embedding_dim, true);
        if (neg_sim > max_neg_sim) {
            max_neg_sim = neg_sim;
            hardest = int(n);
        }
    }
    
    float neg_dist = 1.0 - max_neg_sim;
    
    // Triplet loss: max(0, pos_dist - neg_dist + margin)
    float loss = max(0.0, pos_dist - neg_dist + margin);
    
    // Store results
    losses[batch_idx] = loss;
    hardest_idx[batch_idx] = hardest;
    pos_dists[batch_idx] = pos_dist;
    neg_dists[batch_idx] = neg_dist;
}
