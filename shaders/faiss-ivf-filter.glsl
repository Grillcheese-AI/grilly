#version 450

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Query vector (dim)
layout(set = 0, binding = 0) readonly buffer Query {
    float query[];
};

// Database vectors (num_database, dim)
layout(set = 0, binding = 1) readonly buffer Database {
    float database[];
};

// IVF cluster assignments (num_database) - which cluster each vector belongs to
layout(set = 0, binding = 2) readonly buffer ClusterAssignments {
    uint cluster_ids[];
};

// Query cluster (single value - nearest cluster to query)
layout(set = 0, binding = 3) readonly buffer QueryCluster {
    uint query_cluster_id[];
};

// Candidate indices output (dynamic size)
layout(set = 0, binding = 4) buffer CandidateIndices {
    uint candidates[];
};

// Candidate count
layout(set = 0, binding = 5) buffer CandidateCount {
    uint count[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint num_database;
    uint dim;
    uint num_probe;      // Number of clusters to probe (1 for exact IVF)
};

void main() {
    uint db_idx = gl_GlobalInvocationID.x;
    
    if (db_idx >= num_database) {
        return;
    }
    
    uint query_cluster = query_cluster_id[0];
    uint vec_cluster = cluster_ids[db_idx];
    
    // Check if this vector belongs to query's cluster
    // For IVF with nprobe=1, only search vectors in same cluster
    if (vec_cluster == query_cluster) {
        uint idx = atomicAdd(count[0], 1);
        candidates[idx] = db_idx;
    }
}
