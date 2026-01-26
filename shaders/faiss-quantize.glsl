#version 450

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Input vectors (num_vectors, dim)
layout(set = 0, binding = 0) readonly buffer InputVectors {
    float vectors[];
};

// Codebook centroids (num_centroids, dim)
layout(set = 0, binding = 1) readonly buffer Centroids {
    float centroids[];
};

// Quantized codes output (num_vectors)
layout(set = 0, binding = 2) buffer QuantizedCodes {
    uint codes[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint num_vectors;
    uint num_centroids;
    uint dim;
};

void main() {
    uint vec_idx = gl_GlobalInvocationID.x;
    
    if (vec_idx >= num_vectors) {
        return;
    }
    
    // Find nearest centroid for this vector
    float min_dist = 1e10;
    uint nearest_centroid = 0;
    
    for (uint c = 0; c < num_centroids; c++) {
        float dist = 0.0;
        
        for (uint d = 0; d < dim; d++) {
            uint v_idx = vec_idx * dim + d;
            uint c_idx = c * dim + d;
            float diff = vectors[v_idx] - centroids[c_idx];
            dist += diff * diff;
        }
        
        if (dist < min_dist) {
            min_dist = dist;
            nearest_centroid = c;
        }
    }
    
    codes[vec_idx] = nearest_centroid;
}
