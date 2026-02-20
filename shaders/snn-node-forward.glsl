#version 450

/*
 * SNN Node Forward Pass — Batched neuron step
 *
 * Computes charge-fire-reset for IF, LIF, and PLIF neurons.
 * Supports arbitrary batch+feature shapes (flattened to n_elements).
 *
 * Bindings:
 *   0: input X        [n_elements] (read)
 *   1: V_mem          [n_elements] (read/write — membrane potential)
 *   2: spikes S       [n_elements] (write — output spikes)
 *   3: H              [n_elements] (write — pre-spike membrane, for backward)
 *   4: tau_param      [1 or n_elements] (read — for PLIF, per-neuron tau)
 *
 * Push constants:
 *   n_elements   (uint)
 *   neuron_type  (uint)  0=IF, 1=LIF, 2=PLIF
 *   tau          (float) membrane time constant (LIF/PLIF default)
 *   v_threshold  (float)
 *   v_reset      (float) use -1e9 to signal soft reset
 *   reset_mode   (uint)  0=hard, 1=soft
 *   decay_input  (uint)  0=full input (practical), 1=decay input (physics)
 */

layout(local_size_x = 256) in;

layout(set = 0, binding = 0) readonly buffer InputBuf  { float x_in[]; };
layout(set = 0, binding = 1) buffer VMembuf             { float v_mem[]; };
layout(set = 0, binding = 2) writeonly buffer SpikeBuf  { float spikes[]; };
layout(set = 0, binding = 3) writeonly buffer HBuf      { float h_out[]; };
layout(set = 0, binding = 4) readonly buffer TauBuf     { float tau_param[]; };

layout(push_constant) uniform PushConstants {
    uint  n_elements;
    uint  neuron_type;   // 0=IF, 1=LIF, 2=PLIF
    float tau;
    float v_threshold;
    float v_reset;
    uint  reset_mode;    // 0=hard, 1=soft
    uint  decay_input;   // 0=full input, 1=input/tau
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= n_elements) return;

    float v = v_mem[idx];
    float inp = x_in[idx];
    float h;

    // Charge phase
    if (neuron_type == 0u) {
        // IF: H = V + X (no leak, no decay)
        h = v + inp;
    } else if (neuron_type == 1u) {
        // LIF neuron
        float decay = 1.0 - 1.0 / tau;
        if (decay_input == 1u) {
            // Physics-accurate: H = V*(1-1/tau) + X/tau
            h = v + (inp - (v - v_reset)) / tau;
        } else {
            // Practical (default): H = V*(1-1/tau) + X
            h = decay * (v - v_reset) + v_reset + inp;
        }
    } else {
        // PLIF: same as LIF but tau read per-neuron from buffer
        float t = max(tau_param[idx], 1.0);
        float decay = 1.0 - 1.0 / t;
        if (decay_input == 1u) {
            h = v + (inp - (v - v_reset)) / t;
        } else {
            h = decay * (v - v_reset) + v_reset + inp;
        }
    }

    h_out[idx] = h;

    // Fire phase: Heaviside(h - v_threshold)
    float spike = (h >= v_threshold) ? 1.0 : 0.0;
    spikes[idx] = spike;

    // Reset phase
    if (reset_mode == 0u) {
        // Hard reset: V = H * (1 - S) + v_reset * S
        v_mem[idx] = h * (1.0 - spike) + v_reset * spike;
    } else {
        // Soft reset: V = H - v_threshold * S
        v_mem[idx] = h - v_threshold * spike;
    }
}
