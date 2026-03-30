/// moqe-gumbel-router.glsl — Differentiable 2-expert routing via Gumbel-Softmax.
///
/// Replaces hard argmax routing (which kills gradient to unselected expert).
/// Outputs continuous blending weights [0,1] for both experts.
/// Temperature annealing: start at 1.0, decay to 0.01 during training.
///
/// At low temperature: softmax hardens to argmax (discrete routing).
/// At high temperature: both experts get signal (smooth gradient flow).

#version 450
#extension GL_KHR_shader_subgroup_basic : require

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(std430, set = 0, binding = 0) readonly buffer LogitBuf  { vec2 RouterLogits[];  };
layout(std430, set = 0, binding = 1) writeonly buffer WeightBuf { vec2 ExpertWeights[]; };

layout(push_constant) uniform Params {
    uint seq_len;
    float temperature;
    uint seed;
} p;

uint pcg_hash(uint state) {
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float random_float(inout uint state) {
    state = pcg_hash(state);
    return float(state) / 4294967296.0;
}

float gumbel_noise(inout uint state) {
    float u = max(random_float(state), 1e-7);
    return -log(-log(u));
}

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= p.seq_len) return;

    uint rng_state = idx ^ pcg_hash(p.seed);

    vec2 logits = RouterLogits[idx];

    vec2 noisy_logits;
    noisy_logits.x = logits.x + gumbel_noise(rng_state);
    noisy_logits.y = logits.y + gumbel_noise(rng_state);

    float max_logit = max(noisy_logits.x, noisy_logits.y);
    float exp_x = exp((noisy_logits.x - max_logit) / p.temperature);
    float exp_y = exp((noisy_logits.y - max_logit) / p.temperature);
    float sum_exp = exp_x + exp_y;

    ExpertWeights[idx] = vec2(exp_x / sum_exp, exp_y / sum_exp);
}
