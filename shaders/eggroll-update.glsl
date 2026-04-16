#version 450

/*
 * EGGROLL Fused Weight Update & Merit Modulation
 *
 * Implements the rank-r merit-modulated update:
 *   W_new = W_orig + (sum_i w_i * (u_i @ v_i.T)) * Merit
 *   Merit_new = Merit * (increase if update > eps else decay)
 *
 * Parallelizes over (d_out, d_in).
 */

layout(local_size_x = 16, local_size_y = 16) in;

layout(binding = 0) buffer WeightBuffer { float W[];      }; // Dequantized weights (or float)
layout(binding = 1) buffer MeritBuffer  { float Merit[];  }; 
layout(binding = 2) readonly buffer UBuffer { float U_pool[]; }; // (d_out, n_workers)
layout(binding = 3) readonly buffer VBuffer { float V_pool[]; }; // (d_in, n_workers)
layout(binding = 4) readonly buffer TopIdx  { uint  Indices[]; };
layout(binding = 5) readonly buffer TopFit  { float Weights[]; };

layout(push_constant) uniform PushConstants {
    uint d_out;
    uint d_in;
    uint top_k;
    uint n_workers;
    float merit_increase;
    float merit_decay;
} params;

void main() {
    uint i = gl_GlobalInvocationID.y; // Row
    uint j = gl_GlobalInvocationID.x; // Col

    if (i >= params.d_out || j >= params.d_in) return;

    uint idx = i * params.d_in + j;
    
    float w_orig = W[idx];
    float m_orig = Merit[idx];
    
    // Compute weighted average delta: sum_k w_k * u_k[i] * v_k[j]
    float avg_delta = 0.0;
    for (uint k = 0; k < params.top_k; ++k) {
        uint worker_idx = Indices[k];
        float fw = Weights[k];
        
        float uk = U_pool[i * params.n_workers + worker_idx];
        float vk = V_pool[j * params.n_workers + worker_idx];
        
        avg_delta += fw * (uk * vk);
    }
    
    // Merit-modulated update
    float delta_m = avg_delta * m_orig;
    W[idx] = w_orig + delta_m;
    
    // Update merit
    if (abs(avg_delta) > 1e-8) {
        Merit[idx] *= params.merit_increase;
    } else {
        Merit[idx] *= params.merit_decay;
    }
}
