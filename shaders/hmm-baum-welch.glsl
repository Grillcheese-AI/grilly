#version 450
#extension GL_EXT_shader_atomic_float : enable

// ═══════════════════════════════════════════════════════════════════════
// HMM Baum-Welch E-step Shader — Transition count accumulation
//
// Computes xi[t][i][j] = alpha[t][i] * A[i][j] * B[t+1][j] * beta[t+1][j] / P(O)
// and atomically accumulates into A_num[i][j] += xi[t][i][j]
//
// Dispatch: (n_states * n_states * (T-1)) / 256 workgroups
// Each thread computes one (t, i, j) triple.
//
// Requires alpha, beta, log_A, log_B pre-computed.
// Uses atomicAdd(float) for concurrent accumulation.
// ═══════════════════════════════════════════════════════════════════════

layout(local_size_x = 256) in;

layout(push_constant) uniform PushConstants {
    uint n_states;
    uint T;
    float log_likelihood;  // log P(O | lambda) for normalization
};

// Pre-computed forward/backward log-probabilities
layout(std430, binding = 0) readonly buffer LogAlpha {
    float log_alpha[];  // [T * n_states]
};

layout(std430, binding = 1) readonly buffer LogBeta {
    float log_beta[];   // [T * n_states]
};

layout(std430, binding = 2) readonly buffer LogA {
    float log_A[];      // [n_states * n_states]
};

layout(std430, binding = 3) readonly buffer LogB {
    float log_B[];      // [T * n_states]
};

// Accumulation buffers (atomically updated)
layout(std430, binding = 4) buffer ANum {
    float A_num[];      // [n_states * n_states]
};

layout(std430, binding = 5) buffer PiNum {
    float pi_num[];     // [n_states]
};

void main() {
    uint tid = gl_GlobalInvocationID.x;

    // Map thread to (t, i, j) triple
    uint n2 = n_states * n_states;
    uint total = n2 * (T - 1);

    if (tid >= total) return;

    uint t = tid / n2;
    uint remainder = tid % n2;
    uint i = remainder / n_states;
    uint j = remainder % n_states;

    // xi[t][i][j] = exp(log_alpha[t][i] + log_A[i][j] + log_B[t+1][j] + log_beta[t+1][j] - log_likelihood)
    float log_xi = log_alpha[t * n_states + i]
                 + log_A[i * n_states + j]
                 + log_B[(t + 1) * n_states + j]
                 + log_beta[(t + 1) * n_states + j]
                 - log_likelihood;

    float xi_val = exp(log_xi);

    // Atomically accumulate transition counts
    atomicAdd(A_num[i * n_states + j], xi_val);

    // Accumulate initial state counts from gamma[0] (only t=0 threads, i==j diagonal)
    if (t == 0 && i == j) {
        float log_gamma_0 = log_alpha[i] + log_beta[i] - log_likelihood;
        atomicAdd(pi_num[i], exp(log_gamma_0));
    }
}
