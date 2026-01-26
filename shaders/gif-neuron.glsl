#version 450

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Input current for each neuron
layout(set = 0, binding = 0) readonly buffer InputCurrent {
    float I_input[];
};

// Membrane potential state
layout(set = 0, binding = 1) buffer MembranePotential {
    float V_mem[];
};

// Adaptation current state (slow dynamics)
layout(set = 0, binding = 2) buffer AdaptationCurrent {
    float I_adapt[];
};

// Input gate state (controls how much input affects membrane)
layout(set = 0, binding = 3) buffer InputGate {
    float g_input[];
};

// Forget gate state (controls membrane leak)
layout(set = 0, binding = 4) buffer ForgetGate {
    float g_forget[];
};

// Refractory counter state
layout(set = 0, binding = 5) buffer RefractoryState {
    float t_refrac[];
};

// Output spikes
layout(set = 0, binding = 6) buffer Spikes {
    float spikes[];
};

// Previous spike time (for temporal gating)
layout(set = 0, binding = 7) buffer LastSpikeTime {
    float t_last_spike[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint n_neurons;
    float dt;                 // Time step
    float current_time;       // Current simulation time
    
    // Membrane dynamics
    float tau_mem;            // Membrane time constant
    float V_rest;             // Resting potential
    float V_reset;            // Reset potential
    float V_thresh;           // Spike threshold
    float R_mem;              // Membrane resistance
    
    // Adaptation parameters
    float tau_adapt;          // Adaptation time constant
    float delta_adapt;        // Adaptation increment per spike
    float b_adapt;            // Adaptation coupling strength
    
    // Gating parameters
    float tau_gate;           // Gate time constant
    float gate_strength;      // How strongly gates modulate dynamics
    
    // Refractory period
    float t_refrac_period;
};

// Sigmoid activation for gates
float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

void main() {
    uint gID = gl_GlobalInvocationID.x;
    
    if (gID >= n_neurons) {
        return;
    }
    
    // Read current state
    float V = V_mem[gID];
    float I_a = I_adapt[gID];
    float g_i = g_input[gID];
    float g_f = g_forget[gID];
    float t_ref = t_refrac[gID];
    float I = I_input[gID];
    float t_last = t_last_spike[gID];
    
    float spike = 0.0;
    
    // Check refractory period
    if (t_ref > 0.0) {
        t_ref -= dt;
        t_ref = max(t_ref, 0.0);
        V = V_reset;
    } else {
        // Update gates based on input and recent spiking history
        float dt_since_spike = current_time - t_last;
        
        // Input gate: controls how much input current affects membrane
        // Higher when input is strong and neuron hasn't spiked recently
        float input_gate_target = sigmoid(gate_strength * (I - 0.5 * I_a));
        float recent_spike_suppression = exp(-dt_since_spike / tau_mem);
        input_gate_target *= (1.0 - 0.5 * recent_spike_suppression);
        
        // Forget gate: controls membrane leak
        // Higher forget = more leak (closer to standard LIF)
        // Lower forget = more memory retention
        float forget_gate_target = sigmoid(gate_strength * (0.5 - abs(V - V_rest) / V_thresh));
        
        // Update gates with exponential moving average
        float alpha_gate = dt / tau_gate;
        alpha_gate = clamp(alpha_gate, 0.0, 1.0);
        g_i = g_i + alpha_gate * (input_gate_target - g_i);
        g_f = g_f + alpha_gate * (forget_gate_target - g_f);
        
        // GIF dynamics: gated integration
        // dV/dt = g_f * (-(V - V_rest) / tau) + g_i * R * I - b * I_a
        float leak_term = g_f * (-(V - V_rest) / tau_mem);
        float input_term = g_i * R_mem * I;
        float adapt_term = -b_adapt * I_a;
        
        float dV = leak_term + input_term + adapt_term;
        V += dt * dV;
        
        // Update adaptation current (slow negative feedback)
        // dI_a/dt = -I_a / tau_adapt
        float dI_a = -I_a / tau_adapt;
        I_a += dt * dI_a;
        
        // Check for spike
        if (V >= V_thresh) {
            spike = 1.0;
            V = V_reset;
            t_ref = t_refrac_period;
            t_last = current_time;
            
            // Increment adaptation on spike
            I_a += delta_adapt;
        }
    }
    
    // Write updated state
    V_mem[gID] = V;
    I_adapt[gID] = I_a;
    g_input[gID] = g_i;
    g_forget[gID] = g_f;
    t_refrac[gID] = t_ref;
    spikes[gID] = spike;
    t_last_spike[gID] = t_last;
}
