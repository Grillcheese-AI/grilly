#version 450

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Input spikes (batch_size, seq_len, features)
layout(set = 0, binding = 0) readonly buffer Spikes {
    float spikes[];
};

// Previous trace state
layout(set = 0, binding = 1) readonly buffer PreviousTrace {
    float prev_trace[];
};

// Updated trace output
layout(set = 0, binding = 2) buffer NewTrace {
    float new_trace[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint batch_size;
    uint seq_len;
    uint features;
    float trace_decay;    // Exponential decay (e.g., 0.95)
};

void main() {
    // Each thread handles one (batch, feature) pair
    uint gID = gl_GlobalInvocationID.x;
    uint total_elements = batch_size * features;
    
    if (gID >= total_elements) {
        return;
    }
    
    uint batch_idx = gID / features;
    uint feat_idx = gID % features;
    
    // Compute average spike activity over sequence
    float spike_activity = 0.0;
    
    for (uint t = 0; t < seq_len; t++) {
        uint spike_idx = batch_idx * seq_len * features + t * features + feat_idx;
        spike_activity += spikes[spike_idx];
    }
    
    spike_activity /= float(seq_len);
    
    // Update trace with exponential decay
    // trace = decay * prev_trace + (1 - decay) * activity
    float prev = prev_trace[gID];
    float updated = trace_decay * prev + (1.0 - trace_decay) * spike_activity;
    
    new_trace[gID] = updated;
}
