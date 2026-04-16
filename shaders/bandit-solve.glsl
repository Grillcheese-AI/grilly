#version 450

/*
 * Multi-Armed Bandit Top-2 Solver (Track-and-Stop)
 *
 * Parallelizes over 'n_instances'. Each instance has 'K' arms.
 * Computes optimal sampling proportions (TargetW) and stopping criterion.
 *
 * Inputs:
 *   MuHat: (K, n_instances) - Estimated means
 *   N:     (K, n_instances) - Sample counts
 *
 * Outputs:
 *   TargetW: (K, n_instances) - Optimal proportions
 *   Stop:    (n_instances)    - Boolean (or 1/0) stopping flag
 *
 * Distribution: Gaussian (KL(p||q) = (p-q)^2 / 2)
 */

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0) readonly  buffer MuBuffer     { float MuHat[]; };
layout(binding = 1) readonly  buffer NBuffer      { float N[];     };
layout(binding = 2) writeonly buffer WBuffer      { float TargetW[]; };
layout(binding = 3) writeonly buffer StopBuffer   { uint  StopFlags[]; };

layout(push_constant) uniform PushConstants {
    uint n_arms;
    uint n_instances;
    uint iters;
    float delta;
} params;

// Maximum arms supported in a single thread's stack. 
// For larger K, we would need to redesign to shared memory.
#define MAX_ARMS 128

float kl_gaussian(float mu, float nu) {
    float diff = mu - nu;
    return (diff * diff) * 0.5;
}

void main() {
    uint inst = gl_GlobalInvocationID.x;
    if (inst >= params.n_instances) return;

    uint K = params.n_arms;
    if (K > MAX_ARMS) return; // TODO: Fallback or error

    float mu[MAX_ARMS];
    float counts[MAX_ARMS];
    float w[MAX_ARMS];
    
    uint best_idx = 0;
    float mu_best = -1e30;
    float total_n = 0.0;

    // Load Mu and N, find best arm
    for (uint k = 0; k < K; ++k) {
        uint idx = k * params.n_instances + inst;
        mu[k] = MuHat[idx];
        counts[k] = N[idx];
        w[k] = 1.0 / float(K);
        total_n += counts[k];
        
        if (mu[k] > mu_best) {
            mu_best = mu[k];
            best_idx = k;
        }
    }

    // --- Top-2 Algorithm for Optimal Proportions ---
    for (uint i = 1; i < params.iters; ++i) {
        float sum_kr = 0.0;
        
        // Precompute midpoints and KL ratios relative to best
        for (uint k = 0; k < K; ++k) {
            if (k == best_idx) continue;
            
            // Midpoint: (w_k * mu_k + w_best * mu_best) / (w_k + w_best)
            float mu_avg = (w[k] * mu[k] + w[best_idx] * mu_best) / (w[k] + w[best_idx] + 1e-15);
            
            float kl_best = kl_gaussian(mu_best, mu_avg);
            float kl_k    = kl_gaussian(mu[k], mu_avg);
            
            sum_kr += kl_best / (kl_k + 1e-15);
        }

        if (sum_kr > 1.0) {
            // Case 1: Best arm is undersampled relative to sum of others
            w[best_idx] += 1.0;
        } else {
            // Case 2: Find the arm k != best that minimizes the KL objective
            uint min_arm = 0;
            float min_val = 1e30;
            
            for (uint k = 0; k < K; ++k) {
                if (k == best_idx) continue;
                
                float mu_avg = (w[k] * mu[k] + w[best_idx] * mu_best) / (w[k] + w[best_idx] + 1e-15);
                float val = w[best_idx] * kl_gaussian(mu_best, mu_avg) + w[k] * kl_gaussian(mu[k], mu_avg);
                
                if (val < min_val) {
                    min_val = val;
                    min_arm = k;
                }
            }
            w[min_arm] += 1.0;
        }
    }

    // Finalize W (normalize by total iterations)
    for (uint k = 0; k < K; ++k) {
        uint idx = k * params.n_instances + inst;
        TargetW[idx] = w[k] / float(params.iters);
    }

    // --- Stopping Criterion (GLRT) ---
    uint stop = 0;
    if (total_n > 0.0) {
        float min_glrt = 1e30;
        for (uint k = 0; k < K; ++k) {
            if (k == best_idx) continue;
            
            float wk = counts[k] / total_n;
            float wbest = counts[best_idx] / total_n;
            
            float mu_avg = (wk * mu[k] + wbest * mu_best) / (wk + wbest + 1e-15);
            float obj = wbest * kl_gaussian(mu_best, mu_avg) + wk * kl_gaussian(mu[k], mu_avg);
            float glrt_k = total_n * obj;
            
            min_glrt = min(min_glrt, glrt_k);
        }
        
        // beta(N, delta) = log(K-1) - log(delta) + log(1 + log(total_n))
        float threshold = log(float(K) - 1.0) - log(params.delta) + log(1.0 + log(max(total_n, 1.0)));
        if (min_glrt >= threshold) {
            stop = 1;
        }
    }
    
    // Check for forced round-robin (any N == 0)
    for (uint k = 0; k < K; ++k) {
        if (counts[k] == 0.0) {
            stop = 0;
            break;
        }
    }

    StopFlags[inst] = stop;
}
