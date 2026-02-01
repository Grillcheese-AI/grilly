/**
 * LoRA Forward Pass Shader
 * 
 * Computes: output = x @ W^T + scale * (x @ A^T @ B^T)
 * 
 * This is a fused operation that combines:
 * 1. Base linear transformation (x @ W^T)
 * 2. LoRA adaptation (scale * x @ A^T @ B^T)
 * 
 * Memory layout:
 * - x: (batch, in_features) - Input tensor
 * - W: (out_features, in_features) - Frozen base weights
 * - A: (rank, in_features) - LoRA A matrix
 * - B: (out_features, rank) - LoRA B matrix
 * - output: (batch, out_features) - Output tensor
 */

#version 450

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

// Input tensor: (batch, in_features)
layout(std430, binding = 0) readonly buffer InputBuffer {
    float input_data[];
};

// Base weights: (out_features, in_features) - frozen
layout(std430, binding = 1) readonly buffer WeightBuffer {
    float weight_data[];
};

// LoRA A matrix: (rank, in_features)
layout(std430, binding = 2) readonly buffer LoraABuffer {
    float lora_a_data[];
};

// LoRA B matrix: (out_features, rank)
layout(std430, binding = 3) readonly buffer LoraBBuffer {
    float lora_b_data[];
};

// Output tensor: (batch, out_features)
layout(std430, binding = 4) writeonly buffer OutputBuffer {
    float output_data[];
};

// Intermediate buffer for x @ A^T: (batch, rank)
layout(std430, binding = 5) buffer IntermediateBuffer {
    float intermediate_data[];
};

// Push constants for dimensions and scaling
layout(push_constant) uniform PushConstants {
    uint batch_size;
    uint in_features;
    uint out_features;
    uint rank;
    float scale;  // alpha / rank
    uint phase;   // 0 = compute x @ A^T, 1 = compute output
} params;

// Shared memory for tiled matrix multiplication
shared float tile_x[16][16];
shared float tile_w[16][16];

void main() {
    uint row = gl_GlobalInvocationID.y;  // batch index
    uint col = gl_GlobalInvocationID.x;  // output feature index
    
    if (params.phase == 0) {
        // Phase 0: Compute x @ A^T -> intermediate (batch, rank)
        if (row >= params.batch_size || col >= params.rank) {
            return;
        }
        
        float sum = 0.0;
        uint num_tiles = (params.in_features + 15) / 16;
        
        for (uint t = 0; t < num_tiles; t++) {
            uint tile_col = t * 16 + gl_LocalInvocationID.x;
            
            // Load input tile
            if (row < params.batch_size && tile_col < params.in_features) {
                tile_x[gl_LocalInvocationID.y][gl_LocalInvocationID.x] = 
                    input_data[row * params.in_features + tile_col];
            } else {
                tile_x[gl_LocalInvocationID.y][gl_LocalInvocationID.x] = 0.0;
            }
            
            // Load A^T tile (A is (rank, in_features), so A^T is (in_features, rank))
            uint a_row = t * 16 + gl_LocalInvocationID.y;
            if (a_row < params.in_features && col < params.rank) {
                tile_w[gl_LocalInvocationID.y][gl_LocalInvocationID.x] = 
                    lora_a_data[col * params.in_features + a_row];
            } else {
                tile_w[gl_LocalInvocationID.y][gl_LocalInvocationID.x] = 0.0;
            }
            
            barrier();
            
            for (uint k = 0; k < 16; k++) {
                sum += tile_x[gl_LocalInvocationID.y][k] * tile_w[k][gl_LocalInvocationID.x];
            }
            
            barrier();
        }
        
        intermediate_data[row * params.rank + col] = sum;
        
    } else {
        // Phase 1: Compute output = x @ W^T + scale * intermediate @ B^T
        if (row >= params.batch_size || col >= params.out_features) {
            return;
        }
        
        float base_sum = 0.0;
        float lora_sum = 0.0;
        
        // Compute base output: x @ W^T
        uint num_tiles_base = (params.in_features + 15) / 16;
        
        for (uint t = 0; t < num_tiles_base; t++) {
            uint tile_col = t * 16 + gl_LocalInvocationID.x;
            
            if (row < params.batch_size && tile_col < params.in_features) {
                tile_x[gl_LocalInvocationID.y][gl_LocalInvocationID.x] = 
                    input_data[row * params.in_features + tile_col];
            } else {
                tile_x[gl_LocalInvocationID.y][gl_LocalInvocationID.x] = 0.0;
            }
            
            uint w_row = t * 16 + gl_LocalInvocationID.y;
            if (w_row < params.in_features && col < params.out_features) {
                tile_w[gl_LocalInvocationID.y][gl_LocalInvocationID.x] = 
                    weight_data[col * params.in_features + w_row];
            } else {
                tile_w[gl_LocalInvocationID.y][gl_LocalInvocationID.x] = 0.0;
            }
            
            barrier();
            
            for (uint k = 0; k < 16; k++) {
                base_sum += tile_x[gl_LocalInvocationID.y][k] * tile_w[k][gl_LocalInvocationID.x];
            }
            
            barrier();
        }
        
        // Compute LoRA output: intermediate @ B^T
        uint num_tiles_lora = (params.rank + 15) / 16;
        
        for (uint t = 0; t < num_tiles_lora; t++) {
            uint tile_col = t * 16 + gl_LocalInvocationID.x;
            
            if (row < params.batch_size && tile_col < params.rank) {
                tile_x[gl_LocalInvocationID.y][gl_LocalInvocationID.x] = 
                    intermediate_data[row * params.rank + tile_col];
            } else {
                tile_x[gl_LocalInvocationID.y][gl_LocalInvocationID.x] = 0.0;
            }
            
            uint b_row = t * 16 + gl_LocalInvocationID.y;
            if (b_row < params.rank && col < params.out_features) {
                tile_w[gl_LocalInvocationID.y][gl_LocalInvocationID.x] = 
                    lora_b_data[col * params.rank + b_row];
            } else {
                tile_w[gl_LocalInvocationID.y][gl_LocalInvocationID.x] = 0.0;
            }
            
            barrier();
            
            for (uint k = 0; k < 16; k++) {
                lora_sum += tile_x[gl_LocalInvocationID.y][k] * tile_w[k][gl_LocalInvocationID.x];
            }
            
            barrier();
        }
        
        // Combine: output = base + scale * lora
        output_data[row * params.out_features + col] = base_sum + params.scale * lora_sum;
    }
}
