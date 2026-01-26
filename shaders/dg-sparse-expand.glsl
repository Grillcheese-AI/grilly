#version 450

// Dentate Gyrus Sparse Expansion Shader
// 32D → 128D with top-k sparsification (~2% activation)
// Bio-inspired pattern separation for capsule memory

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Input capsule vectors (batch, 32)
layout(set = 0, binding = 0) readonly buffer Input {
    float input_data[];
};

// Projection weights (32, 128)
layout(set = 0, binding = 1) readonly buffer Weights {
    float W[];
};

// Output sparse vectors (batch, 128)
layout(set = 0, binding = 2) buffer Output {
    float output_data[];
};

// Temporary activations for sorting (batch, 128)
layout(set = 0, binding = 3) buffer Activations {
    float activations[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint batch_size;
    uint input_dim;      // 32
    uint output_dim;     // 128
    uint k;              // Number of active neurons (~3)
    uint pass_type;      // 0 = project, 1 = sparsify, 2 = normalize
};

shared float shared_vals[256];
shared uint shared_idx[256];

void main() {
    uint gID = gl_GlobalInvocationID.x;
    uint lID = gl_LocalInvocationIndex;
    
    if (pass_type == 0) {
        // Pass 0: Project input to high dimension
        // Each thread computes one output element
        uint total_outputs = batch_size * output_dim;
        if (gID >= total_outputs) return;
        
        uint batch_idx = gID / output_dim;
        uint out_idx = gID % output_dim;
        
        // Compute: activations[batch, out] = sum(input[batch, i] * W[i, out])
        float sum = 0.0;
        for (uint i = 0; i < input_dim; i++) {
            uint input_idx = batch_idx * input_dim + i;
            uint w_idx = i * output_dim + out_idx;
            sum += input_data[input_idx] * W[w_idx];
        }
        
        activations[gID] = sum;
        output_data[gID] = 0.0;  // Initialize output to zero
        
    } else if (pass_type == 1) {
        // Pass 1: Top-k sparsification per batch element
        // Each workgroup handles one batch element
        uint batch_idx = gl_WorkGroupID.x;
        if (batch_idx >= batch_size) return;
        
        uint base_idx = batch_idx * output_dim;
        
        // Load activations into shared memory
        if (lID < output_dim) {
            shared_vals[lID] = abs(activations[base_idx + lID]);
            shared_idx[lID] = lID;
        }
        barrier();
        
        // Partial sort to find top-k (bubble sort top-k elements)
        // More efficient than full sort for small k
        for (uint i = 0; i < k; i++) {
            uint max_idx = i;
            float max_val = shared_vals[i];
            
            // Find max in remaining elements (parallel reduction would be faster)
            if (lID == 0) {
                for (uint j = i + 1; j < output_dim; j++) {
                    if (shared_vals[j] > max_val) {
                        max_val = shared_vals[j];
                        max_idx = j;
                    }
                }
                
                // Swap
                if (max_idx != i) {
                    float tmp_val = shared_vals[i];
                    uint tmp_idx = shared_idx[i];
                    shared_vals[i] = shared_vals[max_idx];
                    shared_idx[i] = shared_idx[max_idx];
                    shared_vals[max_idx] = tmp_val;
                    shared_idx[max_idx] = tmp_idx;
                }
            }
            barrier();
        }
        
        // Write top-k values to output
        if (lID < k) {
            uint orig_idx = shared_idx[lID];
            output_data[base_idx + orig_idx] = activations[base_idx + orig_idx];
        }
        
    } else if (pass_type == 2) {
        // Pass 2: L2 normalize each output vector
        // Each workgroup handles one batch element
        uint batch_idx = gl_WorkGroupID.x;
        if (batch_idx >= batch_size) return;
        
        uint base_idx = batch_idx * output_dim;
        
        // Compute sum of squares (parallel reduction)
        float local_sum = 0.0;
        for (uint i = lID; i < output_dim; i += gl_WorkGroupSize.x) {
            float val = output_data[base_idx + i];
            local_sum += val * val;
        }
        shared_vals[lID] = local_sum;
        barrier();
        
        // Reduce
        for (uint stride = gl_WorkGroupSize.x / 2; stride > 0; stride >>= 1) {
            if (lID < stride) {
                shared_vals[lID] += shared_vals[lID + stride];
            }
            barrier();
        }
        
        float norm = sqrt(shared_vals[0] + 1e-8);
        barrier();
        
        // Normalize
        for (uint i = lID; i < output_dim; i += gl_WorkGroupSize.x) {
            output_data[base_idx + i] /= norm;
        }
    }
}
