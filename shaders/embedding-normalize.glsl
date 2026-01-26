#version 450

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Input embeddings (batch, hidden_dim)
layout(set = 0, binding = 0) readonly buffer Input {
    float input_data[];
};

// Output normalized (batch, hidden_dim)
layout(set = 0, binding = 1) buffer Output {
    float output_data[];
};

// Norms buffer (batch,)
layout(set = 0, binding = 2) buffer Norms {
    float norms[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint batch_size;
    uint hidden_dim;
    uint pass_type;  // 0 = compute norms, 1 = normalize
    float eps;       // Small constant for numerical stability
};

shared float shared_sum[256];

void main() {
    uint gID = gl_GlobalInvocationID.x;
    uint lID = gl_LocalInvocationIndex;
    
    if (pass_type == 0) {
        // Pass 0: Compute L2 norms
        // Each workgroup handles one batch element
        uint batch_idx = gl_WorkGroupID.x;
        if (batch_idx >= batch_size) return;
        
        // Sum of squares for this batch element
        float local_sum = 0.0;
        for (uint d = lID; d < hidden_dim; d += gl_WorkGroupSize.x) {
            uint idx = batch_idx * hidden_dim + d;
            float val = input_data[idx];
            local_sum += val * val;
        }
        shared_sum[lID] = local_sum;
        barrier();
        
        // Parallel reduction
        for (uint stride = gl_WorkGroupSize.x / 2; stride > 0; stride >>= 1) {
            if (lID < stride) {
                shared_sum[lID] += shared_sum[lID + stride];
            }
            barrier();
        }
        
        // Store norm (sqrt of sum of squares)
        if (lID == 0) {
            norms[batch_idx] = sqrt(shared_sum[0] + eps);
        }
        
    } else {
        // Pass 1: Normalize
        uint total_elements = batch_size * hidden_dim;
        if (gID >= total_elements) return;
        
        uint batch_idx = gID / hidden_dim;
        float norm = norms[batch_idx];
        
        output_data[gID] = input_data[gID] / norm;
    }
}
