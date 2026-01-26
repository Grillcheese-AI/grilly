#version 450

layout (local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

// Pre-synaptic activations (batch, time, pre_dim)
layout(set = 0, binding = 0) readonly buffer PreActivations {
    float pre[];
};

// Post-synaptic activations (batch, time, post_dim)
layout(set = 0, binding = 1) readonly buffer PostActivations {
    float post[];
};

// Weights to update (post_dim, pre_dim)
layout(set = 0, binding = 2) buffer Weights {
    float W[];
};

// Pre-synaptic traces (batch, pre_dim)
layout(set = 0, binding = 3) buffer PreTrace {
    float pre_trace[];
};

// Post-synaptic traces (batch, post_dim)
layout(set = 0, binding = 4) buffer PostTrace {
    float post_trace[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint batch_size;
    uint time_steps;
    uint pre_dim;
    uint post_dim;
    float lr_potentiation;  // LTP learning rate
    float lr_depression;    // LTD learning rate
    float trace_decay;      // Exponential trace decay
    uint pass_type;         // 0 = update traces, 1 = update weights
};

void main() {
    uint row = gl_GlobalInvocationID.y;
    uint col = gl_GlobalInvocationID.x;
    
    if (pass_type == 0) {
        // Pass 1: Update eligibility traces
        // Pre-synaptic traces
        if (row < batch_size && col < pre_dim) {
            uint trace_idx = row * pre_dim + col;
            float trace = pre_trace[trace_idx];
            
            // Average activity over time
            float avg_activity = 0.0;
            for (uint t = 0; t < time_steps; t++) {
                uint act_idx = row * time_steps * pre_dim + t * pre_dim + col;
                avg_activity += pre[act_idx];
            }
            avg_activity /= float(time_steps);
            
            // Update with decay
            pre_trace[trace_idx] = trace_decay * trace + (1.0 - trace_decay) * avg_activity;
        }
        
        // Post-synaptic traces
        if (row < batch_size && col < post_dim) {
            uint trace_idx = row * post_dim + col;
            float trace = post_trace[trace_idx];
            
            float avg_activity = 0.0;
            for (uint t = 0; t < time_steps; t++) {
                uint act_idx = row * time_steps * post_dim + t * post_dim + col;
                avg_activity += post[act_idx];
            }
            avg_activity /= float(time_steps);
            
            post_trace[trace_idx] = trace_decay * trace + (1.0 - trace_decay) * avg_activity;
        }
        
    } else if (pass_type == 1) {
        // Pass 2: Update weights with STDP
        if (row >= post_dim || col >= pre_dim) return;
        
        // Compute average correlation from traces
        float ltp = 0.0;  // Long-term potentiation
        float ltd = 0.0;  // Long-term depression
        
        for (uint b = 0; b < batch_size; b++) {
            uint pre_idx = b * pre_dim + col;
            uint post_idx = b * post_dim + row;
            
            float pre_t = pre_trace[pre_idx];
            float post_t = post_trace[post_idx];
            
            // STDP rule:
            // LTP: when pre fires before post (pre_trace * post_activity)
            // LTD: when post fires before pre (post_trace * pre_activity)
            ltp += pre_t * post_t;
            ltd += post_t * pre_t;
        }
        
        ltp /= float(batch_size);
        ltd /= float(batch_size);
        
        // Update weight
        uint w_idx = row * pre_dim + col;
        float delta_w = lr_potentiation * ltp - lr_depression * ltd;
        W[w_idx] += delta_w;
    }
}
