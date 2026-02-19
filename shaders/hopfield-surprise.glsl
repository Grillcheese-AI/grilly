#version 450

// Hopfield-Surprise Unified Shader
// Triple dispatch: Hippocampal Surprise → Hopfield attractor → Weight update
//
// Combines sparse episodic recall (CA3) with dense attractor dynamics (Hopfield)
// in a single compute dispatch.  The Hopfield attractor "snaps" noisy hidden
// states to stored prototypes, amplifying needle signals at 128k+ context.
//
// Pipeline:
//   1. Compute instant surprise from hippocampal recall mismatch
//   2. Run Hopfield attractor iterations on surprise-weighted state
//   3. Produce final weight update blending gradient + momentum + attractor

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Current gradients (flattened param, d elements)
layout(set = 0, binding = 0) buffer Gradients {
    float grad[];
};

// CA3-recalled gradient (weighted average, d elements)
layout(set = 0, binding = 1) readonly buffer RecalledGrad {
    float recalled_grad[];
};

// Hopfield attractor patterns (N_patterns × d, row-major)
layout(set = 0, binding = 2) readonly buffer HopfieldPatterns {
    float patterns[];
};

// Hopfield attention scores (N_patterns) — written in pass 0, read in pass 1
layout(set = 0, binding = 3) buffer HopfieldScores {
    float scores[];
};

// Accumulated surprise S_bar (persistent, d elements)
layout(set = 0, binding = 4) buffer SurpriseAccum {
    float s_bar[];
};

// Weight parameters (updated in-place, d elements)
layout(set = 0, binding = 5) buffer Weights {
    float W[];
};

// Per-element update output (d elements)
layout(set = 0, binding = 6) buffer UpdateOutput {
    float delta[];
};

layout(push_constant) uniform PushConsts {
    uint   d;                  // param dimension
    uint   n_patterns;         // number of stored Hopfield patterns
    float  eta_base;           // base learning rate
    float  alpha_momentum;     // surprise EMA decay
    float  lambda_recall;      // hippocampal recall weight
    float  lambda_hopfield;    // Hopfield attractor weight
    float  hopfield_beta;      // Hopfield temperature (sharpness)
    float  surprise_floor;     // minimum surprise
    float  weight_decay;       // L2 regularisation
    float  clip_max;           // gradient clip
    uint   pass_type;          // 0=surprise+scores, 1=hopfield+update
};

void main() {
    uint idx = gl_GlobalInvocationID.x;

    if (pass_type == 0u) {
        // ========== PASS 0: SURPRISE + HOPFIELD SCORE COMPUTATION ==========
        // Each thread handles one element in [0, d)
        if (idx >= d) return;

        float g = grad[idx];
        float r = recalled_grad[idx];

        // Instant surprise: |current - recalled|
        float instant_pe = abs(g - r);

        // Biological momentum
        float s_prev = s_bar[idx];
        float s_new = alpha_momentum * instant_pe + (1.0 - alpha_momentum) * s_prev;
        s_bar[idx] = s_new;

        // Hopfield attention logits: for each pattern p, accumulate
        // score[p] += beta * pattern[p, idx] * (g + lambda * r)
        // (atomicAdd not available in GLSL 450, so we compute per-pattern
        //  scores in a separate partial-sum approach below)
        // NOTE: For simplicity and correctness we compute scores using
        // a dedicated per-pattern thread mapping in pass 1.  Here we just
        // pre-compute the effective gradient stored for pass 1.
        float g_eff = g + lambda_recall * r;
        delta[idx] = g_eff;  // Temporarily store g_eff for pass 1

    } else if (pass_type == 1u) {
        // ========== PASS 1: HOPFIELD REFINEMENT + WEIGHT UPDATE ==========
        if (idx >= d) return;

        float g = grad[idx];
        float r = recalled_grad[idx];
        float g_eff = delta[idx];  // Recover g_eff from pass 0
        float s = s_bar[idx];

        // Hopfield attractor correction: weighted sum of pattern deviations
        // For the CPU fallback, this is a softmax-weighted average of patterns.
        // In the GPU version, patterns are pre-blended by the host.
        // Here we read the pre-blended Hopfield target from scores buffer
        // (the host writes the blended pattern into scores[0..d-1] between passes).
        float hopfield_correction = 0.0;
        if (n_patterns > 0u) {
            hopfield_correction = scores[idx] - g;  // Attractor pull
        }

        // Adaptive LR
        float adaptive_lr = eta_base * (1.0 + max(s, surprise_floor));

        // Final update: gradient + momentum + Hopfield attractor
        float d_val = adaptive_lr * (
            g_eff +
            (s - 1.0) * g +  // Biological momentum term
            lambda_hopfield * hopfield_correction
        );

        // Weight decay
        float w = W[idx];
        d_val += eta_base * weight_decay * w;

        // Clip
        d_val = clamp(d_val, -clip_max, clip_max);

        // Apply
        W[idx] = w - d_val;
        delta[idx] = d_val;
    }
}
