#version 450

// Batch similarity computation: cosine similarity between query and codebook
// Parallel computation for all codebook vectors

layout (local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer Query {
    float query[];
};

layout(set = 0, binding = 1) readonly buffer Codebook {
    float codebook[];  // (codebook_size * dim)
};

layout(set = 0, binding = 2) buffer Similarities {
    float similarities[];
};

layout(push_constant) uniform PushConsts {
    uint dim;
    uint codebook_size;
};

shared float partial_dot[64];
shared float partial_norm_query[64];
shared float partial_norm_vec[64];

void main() {
    uint vec_idx = gl_WorkGroupID.x;
    uint local_idx = gl_LocalInvocationID.x;
    
    if (vec_idx >= codebook_size) {
        return;
    }
    
    // Compute partial dot product and norms
    float dot_sum = 0.0;
    float norm_query_sum = 0.0;
    float norm_vec_sum = 0.0;
    
    for (uint i = local_idx; i < dim; i += 64) {
        float q_val = query[i];
        float v_val = codebook[vec_idx * dim + i];
        
        dot_sum += q_val * v_val;
        norm_query_sum += q_val * q_val;
        norm_vec_sum += v_val * v_val;
    }
    
    partial_dot[local_idx] = dot_sum;
    partial_norm_query[local_idx] = norm_query_sum;
    partial_norm_vec[local_idx] = norm_vec_sum;
    
    barrier();
    
    // Reduce within workgroup
    for (uint stride = 32; stride > 0; stride >>= 1) {
        if (local_idx < stride) {
            partial_dot[local_idx] += partial_dot[local_idx + stride];
            partial_norm_query[local_idx] += partial_norm_query[local_idx + stride];
            partial_norm_vec[local_idx] += partial_norm_vec[local_idx + stride];
        }
        barrier();
    }
    
    // Compute final similarity
    if (local_idx == 0) {
        float dot_product = partial_dot[0];
        float norm_query = sqrt(partial_norm_query[0]);
        float norm_vec = sqrt(partial_norm_vec[0]);
        
        if (norm_query > 0.0 && norm_vec > 0.0) {
            similarities[vec_idx] = dot_product / (norm_query * norm_vec);
        } else {
            similarities[vec_idx] = 0.0;
        }
    }
}
