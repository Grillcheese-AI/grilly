#version 450
#extension GL_EXT_shader_atomic_float : require

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Input vectors (num_vectors, dim)
layout(set = 0, binding = 0) readonly buffer InputVectors {
    float vectors[];
};

// Current centroids (num_centroids, dim)
layout(set = 0, binding = 1) buffer Centroids {
    float centroids[];
};

// Assignments (num_vectors) - which centroid each vector belongs to
layout(set = 0, binding = 2) readonly buffer Assignments {
    uint assignments[];
};

// Cluster counts (num_centroids)
layout(set = 0, binding = 3) buffer ClusterCounts {
    uint counts[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint num_vectors;
    uint num_centroids;
    uint dim;
    uint pass_type;      // 0 = accumulate, 1 = normalize
};

void main() {
    uint gID = gl_GlobalInvocationID.x;
    
    if (pass_type == 0) {
        // Pass 1: Accumulate vectors for each centroid
        if (gID >= num_vectors) return;
        
        uint vec_idx = gID;
        uint centroid_id = assignments[vec_idx];
        
        // Atomic add to centroid (simplified - use separate accumulators in production)
        for (uint d = 0; d < dim; d++) {
            uint v_idx = vec_idx * dim + d;
            uint c_idx = centroid_id * dim + d;
            atomicAdd(centroids[c_idx], vectors[v_idx]);
        }
        
        // Increment cluster count
        atomicAdd(counts[centroid_id], 1);
        
    } else if (pass_type == 1) {
        // Pass 2: Normalize centroids by cluster size
        uint total_elements = num_centroids * dim;
        if (gID >= total_elements) return;
        
        uint centroid_id = gID / dim;
        uint count = counts[centroid_id];
        
        if (count > 0) {
            centroids[gID] /= float(count);
        }
    }
}
