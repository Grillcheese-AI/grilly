#version 450

layout (local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

// Query vectors (num_queries, dim)
layout(set = 0, binding = 0) readonly buffer Queries {
    float queries[];
};

// Database vectors (num_database, dim)
layout(set = 0, binding = 1) readonly buffer Database {
    float database[];
};

// Distance matrix output (num_queries, num_database)
layout(set = 0, binding = 2) buffer Distances {
    float distances[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint num_queries;
    uint num_database;
    uint dim;
    uint distance_type;  // 0 = L2, 1 = cosine, 2 = dot product
};

void main() {
    uint query_idx = gl_GlobalInvocationID.y;
    uint db_idx = gl_GlobalInvocationID.x;
    
    if (query_idx >= num_queries || db_idx >= num_database) {
        return;
    }
    
    float dist = 0.0;
    float q_norm = 0.0;
    float db_norm = 0.0;
    
    if (distance_type == 0) {
        // L2 distance: sqrt(sum((q - db)^2))
        for (uint d = 0; d < dim; d++) {
            uint q_idx = query_idx * dim + d;
            uint d_idx = db_idx * dim + d;
            float diff = queries[q_idx] - database[d_idx];
            dist += diff * diff;
        }
        dist = sqrt(dist);
        
    } else if (distance_type == 1) {
        // Cosine similarity: (q · db) / (||q|| * ||db||)
        // Store as distance: 1 - similarity
        float dot_product = 0.0;
        
        for (uint d = 0; d < dim; d++) {
            uint q_idx = query_idx * dim + d;
            uint d_idx = db_idx * dim + d;
            float q_val = queries[q_idx];
            float db_val = database[d_idx];
            
            dot_product += q_val * db_val;
            q_norm += q_val * q_val;
            db_norm += db_val * db_val;
        }
        
        float similarity = dot_product / (sqrt(q_norm) * sqrt(db_norm) + 1e-8);
        dist = 1.0 - similarity;  // Convert to distance
        
    } else if (distance_type == 2) {
        // Dot product (negative for min-heap compatibility)
        for (uint d = 0; d < dim; d++) {
            uint q_idx = query_idx * dim + d;
            uint d_idx = db_idx * dim + d;
            dist += queries[q_idx] * database[d_idx];
        }
        dist = -dist;  // Negate so smaller is better
    }
    
    uint out_idx = query_idx * num_database + db_idx;
    distances[out_idx] = dist;
}
