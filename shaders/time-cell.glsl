#version 450

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Current time in the sequence
layout(set = 0, binding = 0) readonly buffer CurrentTime {
    float current_time[]; // Single value or per-sequence time
};

// Time cell preferred times (when each cell fires maximally)
layout(set = 0, binding = 1) readonly buffer PreferredTimes {
    float pref_times[]; // One per neuron
};

// Output firing rates
layout(set = 0, binding = 2) buffer FiringRates {
    float rates[];
};

// Optional: membrane potential state for temporal dynamics
layout(set = 0, binding = 3) buffer MembraneState {
    float V_mem[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint n_neurons;           // Number of time cells
    float temporal_width;     // Width of temporal receptive field
    float max_rate;           // Maximum firing rate
    float baseline_rate;      // Baseline firing rate
    float dt;                 // Time step for dynamics
    float tau_adaptation;     // Time constant for adaptation (optional)
    uint use_dynamics;        // 1 = use temporal dynamics, 0 = static tuning curve
};

void main() {
    uint gID = gl_GlobalInvocationID.x;
    
    if (gID >= n_neurons) {
        return;
    }
    
    float t_current = current_time[0];
    float t_preferred = pref_times[gID];
    
    // Calculate temporal distance
    float t_diff = t_current - t_preferred;
    
    if (use_dynamics == 0) {
        // Static Gaussian tuning curve
        float sigma_sq = temporal_width * temporal_width;
        float gaussian = exp(-t_diff * t_diff / (2.0 * sigma_sq));
        float rate = baseline_rate + (max_rate - baseline_rate) * gaussian;
        rates[gID] = rate;
        
    } else {
        // Dynamic sequential activation model
        // Time cells activate in sequence with some temporal dynamics
        
        // Asymmetric temporal receptive field (ramps up, decays down)
        float activation = 0.0;
        
        if (t_diff < 0.0) {
            // Before preferred time: exponential rise
            float rise_rate = 3.0 / temporal_width; // Control rise speed
            activation = exp(rise_rate * t_diff);
        } else {
            // After preferred time: exponential decay
            float decay_rate = 5.0 / temporal_width; // Faster decay than rise
            activation = exp(-decay_rate * t_diff);
        }
        
        // Apply temporal dynamics with membrane potential
        float V = V_mem[gID];
        float target_rate = baseline_rate + (max_rate - baseline_rate) * activation;
        
        // Low-pass filter for smooth transitions
        float alpha = dt / tau_adaptation;
        alpha = clamp(alpha, 0.0, 1.0);
        
        float rate = V + alpha * (target_rate - V);
        
        // Store updated state
        V_mem[gID] = rate;
        rates[gID] = rate;
    }
}
