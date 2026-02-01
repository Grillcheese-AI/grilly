/**
 * LoRA Backward Pass Shader
 * 
 * Computes gradients for LoRA A and B matrices.
 * 
 * Forward: y = x @ W^T + scale * (x @ A^T @ B^T)
 * 
 * Gradients:
 * - grad_B = scale * h^T @ grad_output (where h = x @ A^T)
 * - grad_A = scale * (grad_output @ B)^T @ x
 */

#version 450

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

// Gradient from upstream: (batch, out_features)
layout(std430, binding = 0) readonly buffer GradOutputBuffer {
    float grad_output_data[];
};

// Input from forward pass: (batch, in_features)
layout(std430, binding = 1) readonly buffer InputBuffer {
    float input_data[];
};

// LoRA A matrix: (rank, in_features)
layout(std430, binding = 2) readonly buffer LoraABuffer {
    float lora_a_data[];
};

// LoRA B matrix: (out_features, rank)
layout(std430, binding = 3) readonly buffer LoraBBuffer {
    float lora_b_data[];
};

// Intermediate h = x @ A^T: (batch, rank)
layout(std430, binding = 4) readonly buffer IntermediateBuffer {
    float h_data[];
};

// Gradient for A: (rank, in_features)
layout(std430, binding = 5) buffer GradABuffer {
    float grad_a_data[];
};

// Gradient for B: (out_features, rank)
layout(std430, binding = 6) buffer GradBBuffer {
    float grad_b_data[];
};

// Temporary buffer for grad_output @ B: (batch, rank)
layout(std430, binding = 7) buffer TempBuffer {
    float temp_data[];
};

// Push constants
layout(push_constant) uniform PushConstants {
    uint batch_size;
    uint in_features;
    uint out_features;
    uint rank;
    float scale;
    uint phase;  // 0 = grad_B, 1 = compute temp, 2 = grad_A
} params;

shared float tile_a[16][16];
shared float tile_b[16][16];

void main() {
    uint row = gl_GlobalInvocationID.y;
    uint col = gl_GlobalInvocationID.x;
    
    if (params.phase == 0) {
        // Phase 0: Compute grad_B = scale * h^T @ grad_output
        uint out_idx = row;
        uint rank_idx = col;
        
        if (out_idx >= params.out_features || rank_idx >= params.rank) {
            return;
        }
        
        float sum = 0.0;
        
        for (uint b = 0; b < params.batch_size; b++) {
            float h_val = h_data[b * params.rank + rank_idx];
            float grad_val = grad_output_data[b * params.out_features + out_idx];
            sum += h_val * grad_val;
        }
        
        grad_b_data[out_idx * params.rank + rank_idx] += params.scale * sum;
        
    } else if (params.phase == 1) {
        // Phase 1: Compute temp = grad_output @ B
        uint batch_idx = row;
        uint rank_idx = col;
        
        if (batch_idx >= params.batch_size || rank_idx >= params.rank) {
            return;
        }
        
        float sum = 0.0;
        uint num_tiles = (params.out_features + 15) / 16;
        
        for (uint t = 0; t < num_tiles; t++) {
            uint tile_k = t * 16 + gl_LocalInvocationID.x;
            
            if (batch_idx < params.batch_size && tile_k < params.out_features) {
                tile_a[gl_LocalInvocationID.y][gl_LocalInvocationID.x] = 
                    grad_output_data[batch_idx * params.out_features + tile_k];
            } else {
                tile_a[gl_LocalInvocationID.y][gl_LocalInvocationID.x] = 0.0;
            }
            
            uint b_row = t * 16 + gl_LocalInvocationID.y;
            if (b_row < params.out_features && rank_idx < params.rank) {
                tile_b[gl_LocalInvocationID.y][gl_LocalInvocationID.x] = 
                    lora_b_data[b_row * params.rank + rank_idx];
            } else {
                tile_b[gl_LocalInvocationID.y][gl_LocalInvocationID.x] = 0.0;
            }
            
            barrier();
            
            for (uint k = 0; k < 16; k++) {
                sum += tile_a[gl_LocalInvocationID.y][k] * tile_b[k][gl_LocalInvocationID.x];
            }
            
            barrier();
        }
        
        temp_data[batch_idx * params.rank + rank_idx] = sum;
        
    } else if (params.phase == 2) {
        // Phase 2: Compute grad_A = scale * temp^T @ x
        uint rank_idx = row;
        uint in_idx = col;
        
        if (rank_idx >= params.rank || in_idx >= params.in_features) {
            return;
        }
        
        float sum = 0.0;
        
        for (uint b = 0; b < params.batch_size; b++) {
            float temp_val = temp_data[b * params.rank + rank_idx];
            float x_val = input_data[b * params.in_features + in_idx];
            sum += temp_val * x_val;
        }
        
        grad_a_data[rank_idx * params.in_features + in_idx] += params.scale * sum;
    }
}
