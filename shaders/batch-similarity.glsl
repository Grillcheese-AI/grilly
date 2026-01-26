#version 450

// Batch Cosine Similarity Shader
// Computes cosine similarity between all pairs in a batch
// Used for efficient contrastive learning
//
// Output: similarity matrix (batch, batch) or (batch, num_negatives)

layout (local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

// Input embeddings A (batch_a, dim)
layout(set = 0, binding = 0) readonly buffer InputA {
    float A[];
};

// Input embeddings B (batch_b, dim)
layout(set = 0, binding = 1) readonly buffer InputB {
    float B[];
};

// Pre-computed norms for A (batch_a,)
layout(set = 0, binding = 2) readonly buffer NormsA {
    float norms_a[];
};

// Pre-computed norms for B (batch_b,)
layout(set = 0, binding = 3) readonly buffer NormsB {
    float norms_b[];
};

// Output similarity matrix (batch_a, batch_b)
layout(set = 0, binding = 4) buffer Output {
    float similarities[];
};

layout(push_constant) uniform PushConsts {
    uint batch_a;
    uint batch_b;
    uint dim;
};

// Shared memory for tile-based computation
shared float tile_A[16][16];
shared float tile_B[16][16];

void main() {
    uint row = gl_GlobalInvocationID.y;  // Index in A
    uint col = gl_GlobalInvocationID.x;  // Index in B
    uint lrow = gl_LocalInvocationID.y;
    uint lcol = gl_LocalInvocationID.x;
    
    if (row >= batch_a || col >= batch_b) return;
    
    float dot_product = 0.0;
    
    // Tiled dot product computation
    uint num_tiles = (dim + 15) / 16;
    
    for (uint t = 0; t < num_tiles; t++) {
        uint d_a = t * 16 + lcol;
        uint d_b = t * 16 + lrow;
        
        // Load tiles into shared memory
        if (d_a < dim && row < batch_a) {
            tile_A[lrow][lcol] = A[row * dim + d_a];
        } else {
            tile_A[lrow][lcol] = 0.0;
        }
        
        if (d_b < dim && col < batch_b) {
            tile_B[lrow][lcol] = B[col * dim + d_b];
        } else {
            tile_B[lrow][lcol] = 0.0;
        }
        
        barrier();
        
        // Compute partial dot product
        for (uint k = 0; k < 16; k++) {
            uint d = t * 16 + k;
            if (d < dim) {
                dot_product += tile_A[lrow][k] * tile_B[k][lcol];
            }
        }
        
        barrier();
    }
    
    // Compute cosine similarity
    float norm = norms_a[row] * norms_b[col] + 1e-8;
    float similarity = dot_product / norm;
    
    // Store result
    similarities[row * batch_b + col] = similarity;
}
