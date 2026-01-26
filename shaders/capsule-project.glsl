#version 450

// Capsule Projection Shader
// Projects hidden_dim → capsule_dim and injects cognitive features
// 
// Output capsule structure (32D):
// - dims 0-27: Semantic content (from projection)
// - dims 28-31: Cognitive features (plasticity, consolidation, stability, stress)

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Input embeddings (batch, hidden_dim)
layout(set = 0, binding = 0) readonly buffer Input {
    float input_data[];
};

// Projection weights (capsule_dim, hidden_dim)
layout(set = 0, binding = 1) readonly buffer Weights {
    float W[];
};

// Cognitive features per batch (batch, 4) - plasticity, consolidation, stability, stress
layout(set = 0, binding = 2) readonly buffer CognitiveFeatures {
    float cog_features[];
};

// Output capsules (batch, capsule_dim)
layout(set = 0, binding = 3) buffer Output {
    float output_data[];
};

// Norms buffer (batch,)
layout(set = 0, binding = 4) buffer Norms {
    float norms[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint batch_size;
    uint hidden_dim;
    uint capsule_dim;
    uint semantic_dims;    // 28
    uint cognitive_dims;   // 4
    uint pass_type;        // 0 = project, 1 = blend semantic, 2 = inject cognitive, 3 = normalize
    float semantic_weight; // 0.9
};

shared float shared_sum[256];

void main() {
    uint gID = gl_GlobalInvocationID.x;
    uint lID = gl_LocalInvocationIndex;
    
    if (pass_type == 0) {
        // Pass 0: Project hidden → capsule
        // Each thread computes one output element
        uint total_outputs = batch_size * capsule_dim;
        if (gID >= total_outputs) return;
        
        uint batch_idx = gID / capsule_dim;
        uint cap_idx = gID % capsule_dim;
        
        // Only compute semantic dimensions
        if (cap_idx >= semantic_dims) {
            output_data[gID] = 0.0;
            return;
        }
        
        // Compute: capsule[batch, cap] = sum(input[batch, h] * W[cap, h])
        float sum = 0.0;
        for (uint h = 0; h < hidden_dim; h++) {
            uint input_idx = batch_idx * hidden_dim + h;
            uint w_idx = cap_idx * hidden_dim + h;
            sum += input_data[input_idx] * W[w_idx];
        }
        
        output_data[gID] = sum;
        
    } else if (pass_type == 1) {
        // Pass 1: Normalize semantic portion and compute norm
        // Each workgroup handles one batch element
        uint batch_idx = gl_WorkGroupID.x;
        if (batch_idx >= batch_size) return;
        
        uint base_idx = batch_idx * capsule_dim;
        
        // Compute sum of squares for semantic portion
        float local_sum = 0.0;
        for (uint i = lID; i < semantic_dims; i += gl_WorkGroupSize.x) {
            float val = output_data[base_idx + i];
            local_sum += val * val;
        }
        shared_sum[lID] = local_sum;
        barrier();
        
        // Reduce
        for (uint stride = gl_WorkGroupSize.x / 2; stride > 0; stride >>= 1) {
            if (lID < stride) {
                shared_sum[lID] += shared_sum[lID + stride];
            }
            barrier();
        }
        
        float norm = sqrt(shared_sum[0] + 1e-8);
        if (lID == 0) {
            norms[batch_idx] = norm;
        }
        barrier();
        
        // Normalize semantic portion
        for (uint i = lID; i < semantic_dims; i += gl_WorkGroupSize.x) {
            output_data[base_idx + i] /= norm;
        }
        
    } else if (pass_type == 2) {
        // Pass 2: Blend semantic with cognitive modulation and inject cognitive features
        uint batch_idx = gl_WorkGroupID.x;
        if (batch_idx >= batch_size) return;
        
        uint base_idx = batch_idx * capsule_dim;
        uint cog_base = batch_idx * cognitive_dims;
        
        // Get plasticity for modulation
        float plasticity = cog_features[cog_base];  // First cognitive feature
        
        // Modulate semantic portion: 90% base + 10% * plasticity
        for (uint i = lID; i < semantic_dims; i += gl_WorkGroupSize.x) {
            float val = output_data[base_idx + i];
            output_data[base_idx + i] = val * semantic_weight + val * (1.0 - semantic_weight) * plasticity;
        }
        
        // Inject cognitive features into last dims
        if (lID < cognitive_dims) {
            output_data[base_idx + semantic_dims + lID] = cog_features[cog_base + lID];
        }
        
    } else if (pass_type == 3) {
        // Pass 3: Final L2 normalization of full capsule
        uint batch_idx = gl_WorkGroupID.x;
        if (batch_idx >= batch_size) return;
        
        uint base_idx = batch_idx * capsule_dim;
        
        // Sum of squares
        float local_sum = 0.0;
        for (uint i = lID; i < capsule_dim; i += gl_WorkGroupSize.x) {
            float val = output_data[base_idx + i];
            local_sum += val * val;
        }
        shared_sum[lID] = local_sum;
        barrier();
        
        // Reduce
        for (uint stride = gl_WorkGroupSize.x / 2; stride > 0; stride >>= 1) {
            if (lID < stride) {
                shared_sum[lID] += shared_sum[lID + stride];
            }
            barrier();
        }
        
        float norm = sqrt(shared_sum[0] + 1e-8);
        barrier();
        
        // Normalize
        for (uint i = lID; i < capsule_dim; i += gl_WorkGroupSize.x) {
            output_data[base_idx + i] /= norm;
        }
    }
}
