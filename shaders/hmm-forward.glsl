#version 450
#extension GL_KHR_shader_subgroup_arithmetic : enable

// ═══════════════════════════════════════════════════════════════════════
// HMM Forward Algorithm Shader
//
// Computes alpha[t][j] = B[t][j] * sum_i(alpha[t-1][i] * A[i][j])
// for all states j in parallel, one timestep at a time.
//
// Dispatch: n_states / 256 workgroups, called T-1 times (one per timestep)
// Or: single dispatch with barrier synchronization per timestep.
//
// For n_states <= 256, one workgroup handles all states.
// Each thread computes one state's alpha value.
// ═══════════════════════════════════════════════════════════════════════

layout(local_size_x = 256) in;

layout(push_constant) uniform PushConstants {
    uint n_states;
    uint T;            // Sequence length
    uint current_t;    // Current timestep being computed
};

// Transition matrix A[i][j] = P(s_{t+1}=j | s_t=i)
// Stored as log-probabilities for numerical stability
layout(std430, binding = 0) readonly buffer TransitionMatrix {
    float log_A[];  // [n_states * n_states]
};

// Emission log-probabilities: log B[t][j] = log P(o_t | s_j)
layout(std430, binding = 1) readonly buffer EmissionMatrix {
    float log_B[];  // [T * n_states]
};

// Log-alpha values: log_alpha[t][j]
// Read from t-1, write to t
layout(std430, binding = 2) buffer LogAlpha {
    float log_alpha[];  // [T * n_states]
};

// Shared memory for log-sum-exp reduction
shared float shared_vals[256];

// Numerically stable log-sum-exp
float logsumexp_shared(uint n) {
    // Find max
    float max_val = -1e30;
    for (uint i = 0; i < n; i++) {
        if (shared_vals[i] > max_val) max_val = shared_vals[i];
    }
    if (max_val < -1e29) return -1e30;

    float sum = 0.0;
    for (uint i = 0; i < n; i++) {
        sum += exp(shared_vals[i] - max_val);
    }
    return max_val + log(sum);
}

void main() {
    uint j = gl_GlobalInvocationID.x;  // Target state
    if (j >= n_states) return;

    uint t = current_t;

    if (t == 0) {
        // Initialize: log_alpha[0][j] = log_pi[j] + log_B[0][j]
        // log_pi is stored in the first row of log_A (convention)
        // OR: uniform prior log_pi[j] = -log(n_states)
        float log_pi_j = -log(float(n_states));
        log_alpha[j] = log_pi_j + log_B[j];
        return;
    }

    // Compute log_alpha[t][j] = log_B[t][j] + logsumexp_i(log_alpha[t-1][i] + log_A[i][j])
    // Load all log_alpha[t-1][i] + log_A[i][j] into shared memory
    // Each thread j loads its own column of A

    // For small n_states, single thread does the full sum
    float log_emission = log_B[t * n_states + j];

    // Compute logsumexp over i: log_alpha[t-1][i] + log_A[i][j]
    float max_val = -1e30;
    for (uint i = 0; i < n_states; i++) {
        float v = log_alpha[(t - 1) * n_states + i] + log_A[i * n_states + j];
        if (v > max_val) max_val = v;
    }

    float sum_exp = 0.0;
    if (max_val > -1e29) {
        for (uint i = 0; i < n_states; i++) {
            float v = log_alpha[(t - 1) * n_states + i] + log_A[i * n_states + j];
            sum_exp += exp(v - max_val);
        }
    }

    float log_sum = (max_val > -1e29) ? (max_val + log(sum_exp)) : -1e30;
    log_alpha[t * n_states + j] = log_emission + log_sum;
}
