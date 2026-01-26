#version 450

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Input current for each neuron (batch * time, features)
layout(set = 0, binding = 0) readonly buffer InputCurrent {
    float I_input[];
};

// Attention gains from prosody (batch * time) or (batch * time * features)
layout(set = 0, binding = 1) readonly buffer AttentionGains {
    float attention[];
};

// Membrane potential state
layout(set = 0, binding = 2) buffer MembranePotential {
    float V_mem[];
};

// Adaptive threshold state
layout(set = 0, binding = 3) buffer AdaptiveThreshold {
    float theta[];
};

// Output spikes (batch * time, features)
layout(set = 0, binding = 4) buffer Spikes {
    float spikes[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint batch_size;
    uint seq_len;
    uint features;
    
    // GIF parameters
    float dt;                          // Time step
    float tau;                         // Membrane time constant
    float threshold_base;              // Base threshold
    float alpha;                       // Adaptation rate
    uint L;                            // Multi-bit precision (e.g., 16)
    
    // Prosody modulation
    float attn_mod_strength;           // Attention modulation strength (e.g., 0.3)
    uint attn_is_scalar;               // 1 = one gain per timestep, 0 = per-feature gains
    uint timestep;                     // Current timestep being processed
};

// Multi-bit surrogate gradient approximation
// Returns quantized spike value in [0, L]
float multi_bit_spike(float normalized_v, uint L_bits) {
    // Clamp to reasonable range
    float v_clamped = clamp(normalized_v, 0.0, float(L_bits));
    
    // Quantize to L levels
    float spike_val = floor(v_clamped);
    
    // Ensure in valid range [0, L]
    return clamp(spike_val, 0.0, float(L_bits));
}

void main() {
    uint gID = gl_GlobalInvocationID.x;
    uint batch_elem = batch_size * features;
    
    if (gID >= batch_elem) {
        return;
    }
    
    // Decode: which batch and which feature
    uint batch_idx = gID / features;
    uint feat_idx = gID % features;
    
    // Get current attention gain
    float gain = 1.0;
    if (attn_is_scalar == 1) {
        // One scalar gain per (batch, time)
        uint attn_idx = batch_idx * seq_len + timestep;
        gain = attention[attn_idx];
    } else {
        // Per-feature gains
        uint attn_idx = batch_idx * seq_len * features + timestep * features + feat_idx;
        gain = attention[attn_idx];
    }
    
    // Get input current for this timestep
    uint input_idx = batch_idx * seq_len * features + timestep * features + feat_idx;
    float I = I_input[input_idx];
    
    // Apply attention gain to input (amplify important tokens)
    I = I * gain;
    
    // Read state
    float V = V_mem[gID];
    float theta_current = theta[gID];
    
    // Update membrane potential with decay
    float decay = exp(-dt / tau);
    V = V * decay + I;
    
    // Modulate threshold based on attention
    // High attention → lower threshold (easier to spike)
    float threshold_scale = 1.0 - attn_mod_strength * (gain - 1.0);
    threshold_scale = clamp(threshold_scale, 0.5, 1.5);
    float theta_effective = theta_current * threshold_scale;
    
    // Numerical stability: clamp voltage
    float clamp_limit = float(L) * theta_effective * 2.0;
    V = clamp(V, -clamp_limit, clamp_limit);
    
    // Multi-bit spike generation
    float normalized_v = V / theta_effective;
    float spike = multi_bit_spike(normalized_v, L);
    
    // Soft reset
    V = V - spike * theta_effective;
    
    // Threshold adaptation (modulated by attention)
    // Higher attention → faster adaptation
    if (alpha > 0.0) {
        float alpha_effective = alpha * gain;
        theta_current = theta_current + alpha_effective * spike - alpha_effective * (theta_current - threshold_base);
    }
    
    // Write updated state
    V_mem[gID] = V;
    theta[gID] = theta_current;
    
    // Write output spike
    uint out_idx = batch_idx * seq_len * features + timestep * features + feat_idx;
    spikes[out_idx] = spike;
}
