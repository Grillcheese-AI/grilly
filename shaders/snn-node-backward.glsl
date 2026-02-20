#version 450

/*
 * SNN Node Backward Pass — Surrogate gradient computation
 *
 * Computes grad_x = surrogate_grad(H - v_threshold) * grad_spike
 *
 * Bindings:
 *   0: grad_spike   [n_elements] (read — upstream gradient)
 *   1: h_cache      [n_elements] (read — pre-spike membrane from forward)
 *   2: grad_x       [n_elements] (write — gradient w.r.t. input)
 *
 * Push constants:
 *   n_elements     (uint)
 *   alpha          (float) surrogate function sharpness
 *   surrogate_type (uint)  0=ATan, 1=Sigmoid, 2=FastSigmoid
 *   v_threshold    (float)
 */

layout(local_size_x = 256) in;

layout(set = 0, binding = 0) readonly buffer GradSpikeBuf { float grad_spike[]; };
layout(set = 0, binding = 1) readonly buffer HCacheBuf    { float h_cache[]; };
layout(set = 0, binding = 2) writeonly buffer GradXBuf    { float grad_x[]; };

layout(push_constant) uniform PushConstants {
    uint  n_elements;
    float alpha;
    uint  surrogate_type;  // 0=ATan, 1=Sigmoid, 2=FastSigmoid
    float v_threshold;
};

#define PI 3.14159265358979323846

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= n_elements) return;

    float x = h_cache[idx] - v_threshold;
    float sg;

    if (surrogate_type == 0u) {
        // ATan: alpha / (2 * (1 + (pi*alpha*x/2)^2))
        float pax = PI * alpha * x / 2.0;
        sg = alpha / (2.0 * (1.0 + pax * pax));
    } else if (surrogate_type == 1u) {
        // Sigmoid: alpha * sig(alpha*x) * (1 - sig(alpha*x))
        float s = 1.0 / (1.0 + exp(-alpha * x));
        sg = alpha * s * (1.0 - s);
    } else {
        // FastSigmoid: alpha / (2 * (1 + alpha*|x|)^2)
        float denom = 1.0 + alpha * abs(x);
        sg = alpha / (2.0 * denom * denom);
    }

    grad_x[idx] = grad_spike[idx] * sg;
}
