#version 450

// Surprise Recall Blend Shader
// Blends K recalled gradient episodes by DG similarity into a single recalled
// gradient vector.  Runs before surprise-momentum.glsl.
//
// Input: top-K recalled gradient directions (K × d) and similarity scores (K)
// Output: single weighted-average recalled gradient (d)

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// K recalled gradient directions, packed row-major (K * d)
layout(set = 0, binding = 0) readonly buffer RecalledEpisodes {
    float episodes[];
};

// Similarity scores for each recalled episode (K)
layout(set = 0, binding = 1) readonly buffer Similarities {
    float sims[];
};

// Loss delta (improvement signal) for each recalled episode (K)
layout(set = 0, binding = 2) readonly buffer LossDeltas {
    float loss_deltas[];
};

// Output: blended recalled gradient (d)
layout(set = 0, binding = 3) buffer BlendedOutput {
    float blended[];
};

layout(push_constant) uniform PushConsts {
    uint d;                    // dimension of gradient vectors
    uint K;                    // number of recalled episodes
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= d) {
        return;
    }

    // Weighted average: w_k = max(0, sim_k) * max(0.01, -loss_delta_k + 0.5)
    float sum_val = 0.0;
    float sum_w = 0.0;

    for (uint k = 0; k < K; k++) {
        float sim_k = max(sims[k], 0.0);
        // Negative loss_delta = improvement = good signal
        float outcome_k = max(0.01, -loss_deltas[k] + 0.5);
        float w = sim_k * outcome_k;
        sum_val += w * episodes[k * d + idx];
        sum_w += w;
    }

    if (sum_w > 1e-9) {
        blended[idx] = sum_val / sum_w;
    } else {
        blended[idx] = 0.0;
    }
}
