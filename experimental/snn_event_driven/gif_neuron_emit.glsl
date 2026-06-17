#version 450
// gif-neuron.glsl + event-driven spike-list emission.
// Identical GIF dynamics to the dense original; on spike, atomically appends
// the neuron index to a compact fired list (bindings 8-9) that synapse_scatter
// consumes. Dense spikes[] (binding 6) retained for parity/monitoring.

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer InputCurrent     { float I_input[]; };
layout(set = 0, binding = 1)          buffer MembranePotential { float V_mem[]; };
layout(set = 0, binding = 2)          buffer AdaptationCurrent { float I_adapt[]; };
layout(set = 0, binding = 3)          buffer InputGate         { float g_input[]; };
layout(set = 0, binding = 4)          buffer ForgetGate        { float g_forget[]; };
layout(set = 0, binding = 5)          buffer RefractoryState   { float t_refrac[]; };
layout(set = 0, binding = 6)          buffer Spikes            { float spikes[]; };
layout(set = 0, binding = 7)          buffer LastSpikeTime     { float t_last_spike[]; };
// --- event-driven additions ---
layout(set = 0, binding = 8)          buffer FiredIdx          { uint  fired_idx[]; };
layout(set = 0, binding = 9)          buffer FiredCount        { uint  fired_count[]; }; // zeroed each step

layout(push_constant) uniform PushConsts {
    uint n_neurons;
    float dt;
    float current_time;
    float tau_mem;
    float V_rest;
    float V_reset;
    float V_thresh;
    float R_mem;
    float tau_adapt;
    float delta_adapt;
    float b_adapt;
    float tau_gate;
    float gate_strength;
    float t_refrac_period;
};

float sigmoid(float x) { return 1.0 / (1.0 + exp(-x)); }

void main() {
    uint gID = gl_GlobalInvocationID.x;
    if (gID >= n_neurons) return;

    float V      = V_mem[gID];
    float I_a    = I_adapt[gID];
    float g_i    = g_input[gID];
    float g_f    = g_forget[gID];
    float t_ref  = t_refrac[gID];
    float I      = I_input[gID];
    float t_last = t_last_spike[gID];

    float spike = 0.0;

    if (t_ref > 0.0) {
        t_ref -= dt;
        t_ref = max(t_ref, 0.0);
        V = V_reset;
    } else {
        float dt_since_spike = current_time - t_last;
        float input_gate_target = sigmoid(gate_strength * (I - 0.5 * I_a));
        float recent_spike_suppression = exp(-dt_since_spike / tau_mem);
        input_gate_target *= (1.0 - 0.5 * recent_spike_suppression);
        float forget_gate_target = sigmoid(gate_strength * (0.5 - abs(V - V_rest) / V_thresh));

        float alpha_gate = clamp(dt / tau_gate, 0.0, 1.0);
        g_i = g_i + alpha_gate * (input_gate_target - g_i);
        g_f = g_f + alpha_gate * (forget_gate_target - g_f);

        float leak_term  = g_f * (-(V - V_rest) / tau_mem);
        float input_term = g_i * R_mem * I;
        float adapt_term = -b_adapt * I_a;
        V += dt * (leak_term + input_term + adapt_term);
        I_a += dt * (-I_a / tau_adapt);

        if (V >= V_thresh) {
            spike = 1.0;
            V = V_reset;
            t_ref = t_refrac_period;
            t_last = current_time;
            I_a += delta_adapt;
            // event-driven: append this neuron to the compact fired list
            uint slot = atomicAdd(fired_count[0], 1u);
            fired_idx[slot] = gID;
        }
    }

    V_mem[gID]        = V;
    I_adapt[gID]      = I_a;
    g_input[gID]      = g_i;
    g_forget[gID]     = g_f;
    t_refrac[gID]     = t_ref;
    spikes[gID]       = spike;
    t_last_spike[gID] = t_last;
}
