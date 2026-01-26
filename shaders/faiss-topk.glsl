#version 450

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Distances (num_queries, num_database)
layout(set = 0, binding = 0) readonly buffer Distances {
    float distances[];
};

// Database indices (num_database)
layout(set = 0, binding = 1) readonly buffer DatabaseIndices {
    uint db_indices[];
};

// Top-k indices output (num_queries, k)
layout(set = 0, binding = 2) buffer TopKIndices {
    uint topk_indices[];
};

// Top-k distances output (num_queries, k)
layout(set = 0, binding = 3) buffer TopKDistances {
    float topk_distances[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint num_queries;
    uint num_database;
    uint k;
};

void main() {
    uint query_idx = gl_GlobalInvocationID.x;
    
    if (query_idx >= num_queries) {
        return;
    }

    // Initialize top-k slots for this query to sentinel
    for (uint init = 0; init < k; init++) {
        topk_indices[query_idx * k + init] = uint(-1);
        topk_distances[query_idx * k + init] = 1e10;
    }
    
    // Selection sort to find top-k smallest distances
    // Note: for large k, use heap-based approach
    
    const float MAX_FLOAT = 1e10;
    
    for (uint rank = 0; rank < k; rank++) {
        float min_dist = MAX_FLOAT;
        uint min_idx = 0;
        
        // Find minimum distance not yet selected
        for (uint db_idx = 0; db_idx < num_database; db_idx++) {
            uint dist_idx = query_idx * num_database + db_idx;
            float dist = distances[dist_idx];
            
            // Check if already selected
            bool already_selected = false;
            for (uint prev = 0; prev < rank; prev++) {
                uint prev_idx = query_idx * k + prev;
                if (topk_indices[prev_idx] == db_idx) {
                    already_selected = true;
                    break;
                }
            }
            
            if (!already_selected && dist < min_dist) {
                min_dist = dist;
                min_idx = db_idx;
            }
        }
        
        // Store top-k result
        uint out_idx = query_idx * k + rank;
        topk_indices[out_idx] = min_idx;
        topk_distances[out_idx] = min_dist;
    }
}
