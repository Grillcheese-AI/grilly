/// lsq-stochastic-quant.glsl — Learnable Step Size + Stochastic Rounding.
///
/// Replaces STE round() which causes gradient mismatch.
/// Stochastic rounding: E[quantized] = continuous value (zero bias).
/// LSQ: scale is learnable, gradient normalized by 1/sqrt(N * Q_max).
///
/// vec4 loads for 128-bit bandwidth saturation.

#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(std430, set = 0, binding = 0) readonly buffer InBuf   { vec4 Activations[]; };
layout(std430, set = 0, binding = 1) writeonly buffer OutBuf  { vec4 Quantized[];   };

layout(push_constant) uniform Params {
    uint total_vec4s;
    float scale;
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

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= p.total_vec4s) return;

    uint rng_state = idx ^ pcg_hash(p.seed);

    vec4 x = Activations[idx];

    float inv_scale = 1.0 / p.scale;
    vec4 x_scaled = x * inv_scale;
    vec4 x_floor = floor(x_scaled);
    vec4 fraction = x_scaled - x_floor;

    vec4 u = vec4(
        random_float(rng_state),
        random_float(rng_state),
        random_float(rng_state),
        random_float(rng_state)
    );

    vec4 stoch_mask = vec4(
        u.x < fraction.x ? 1.0 : 0.0,
        u.y < fraction.y ? 1.0 : 0.0,
        u.z < fraction.z ? 1.0 : 0.0,
        u.w < fraction.w ? 1.0 : 0.0
    );

    vec4 x_quantized_int = clamp(x_floor + stoch_mask, -127.0, 127.0);
    Quantized[idx] = x_quantized_int * p.scale;
}
