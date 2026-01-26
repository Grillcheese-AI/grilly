#version 450

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Input current for each neuron
layout(set = 0, binding = 0) readonly buffer InputCurrent {
    float I_input[];
};

// Membrane potential state (read and write)
layout(set = 0, binding = 1) buffer MembranePotential {
    float V_mem[];
};

// Refractory counter state (read and write)
layout(set = 0, binding = 2) buffer RefractoryState {
    float t_refrac[];
};

// Output spike buffer (1.0 if spiked, 0.0 otherwise)
layout(set = 0, binding = 3) buffer Spikes {
    float spikes[];
};

// LIF neuron parameters via push constants
layout(push_constant) uniform PushConsts {
    uint n_neurons;           // Number of neurons to simulate
    float dt;                 // Time step (e.g., 0.001 for 1ms)
    float tau_mem;            // Membrane time constant (e.g., 20ms)
    float V_rest;             // Resting potential (e.g., -70mV)
    float V_reset;            // Reset potential after spike (e.g., -75mV)
    float V_thresh;           // Spike threshold (e.g., -55mV)
    float R_mem;              // Membrane resistance (e.g., 1.0)
    float t_refrac_period;    // Refractory period duration (e.g., 2ms)
};

void main() {
    uint gID = gl_GlobalInvocationID.x;
    
    if (gID >= n_neurons) {
        return;
    }
    
    // Read current state
    float V = V_mem[gID];
    float t_ref = t_refrac[gID];
    float I = I_input[gID];
    
    // Initialize spike output to 0
    float spike = 0.0;
    
    // Check if neuron is in refractory period
    if (t_ref > 0.0) {
        // In refractory period: decay counter, keep V at reset
        t_ref -= dt;
        t_ref = max(t_ref, 0.0);
        V = V_reset;
    } else {
        // Not in refractory period: update membrane potential using LIF dynamics
        // dV/dt = (-(V - V_rest) + R * I) / tau
        // Euler integration: V(t+1) = V(t) + dt * dV/dt
        float dV = (-(V - V_rest) + R_mem * I) / tau_mem;
        V += dt * dV;
        
        // Check for spike
        if (V >= V_thresh) {
            spike = 1.0;
            V = V_reset;
            t_ref = t_refrac_period;
        }
    }
    
    // Write updated state
    V_mem[gID] = V;
    t_refrac[gID] = t_ref;
    spikes[gID] = spike;
}
