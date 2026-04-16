#version 450

/*
 * EGGROLL Perturbation Generator
 *
 * Generates N rank-1 perturbations (U and V vectors) for a target layer.
 * Uses PCG-based PRNG for high-quality random values on GPU.
 *
 * Output:
 *   U_pool: (d_out, n_workers)
 *   V_pool: (d_in,  n_workers)
 */

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0) writeonly buffer UBuffer { float U_pool[]; };
layout(binding = 1) writeonly buffer VBuffer { float V_pool[]; };

layout(push_constant) uniform PushConstants {
    uint d_out;
    uint d_in;
    uint n_workers;
    uint seed;
    float sigma;
} params;

// PCG Random Number Generator
uint pcg_hash(uint seed_val) {
    uint state = seed_val * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Map uint to Normal distribution using Box-Muller
vec2 box_muller(uint seed) {
    uint r1 = pcg_hash(seed);
    uint r2 = pcg_hash(seed + 123456789u);
    
    float f1 = float(r1) / 4294967296.0;
    float f2 = float(r2) / 4294967296.0;
    
    float theta = 6.2831853 * f1;
    float rho = sqrt(-2.0 * log(max(f2, 1e-10)));
    
    return vec2(rho * cos(theta), rho * sin(theta));
}

void main() {
    uint id = gl_GlobalInvocationID.x;
    uint total_elements = (params.d_out + params.d_in) * params.n_workers;
    
    // Each thread generates one pair (or partial) of random normals
    // We'll map id to specific vector and worker
    
    // For simplicity, each worker's vectors are contiguous
    if (id < params.n_workers) {
        // This thread generates ALL vectors for one worker
        // (Slow but simple for now. Better to parallelize across k as well)
        // Let's refine: parallelize across k and worker combined.
    }
    
    // Refined mapping:
    // id = worker_idx * (d_out + d_in) + element_idx
    if (id >= total_elements) return;
    
    uint worker_idx = id / (params.d_out + params.d_in);
    uint element_idx = id % (params.d_out + params.d_in);
    
    uint global_seed = params.seed + id;
    
    // We need Normals. Box-Muller generates 2 per call.
    // To be efficient, we'll only call it every 2 elements.
    float val;
    if (id % 2 == 0) {
        val = box_muller(global_seed).x;
    } else {
        val = box_muller(global_seed - 1).y;
    }
    
    val *= params.sigma;

    if (element_idx < params.d_out) {
        // U vector
        uint out_idx = element_idx * params.n_workers + worker_idx;
        U_pool[out_idx] = val;
    } else {
        // V vector
        uint in_idx = (element_idx - params.d_out) * params.n_workers + worker_idx;
        V_pool[in_idx] = val;
    }
}
