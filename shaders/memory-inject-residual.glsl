#version 450

// Memory Injection Shader
// Injects averaged capsule memory vectors into transformer residual stream
// Applied at specified layers (typically 4-5) for hippocampal-style memory retrieval

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Hidden states (batch, seq_len, hidden_dim)
layout(set = 0, binding = 0) buffer HiddenStates {
    float hidden[];
};

// Averaged memory capsule vectors (batch, capsule_dim)
// Pre-averaged from retrieved memories
layout(set = 0, binding = 1) readonly buffer MemoryCapsules {
    float memory_capsules[];
};

// Injection projection weights (hidden_dim, capsule_dim)
layout(set = 0, binding = 2) readonly buffer InjectionWeights {
    float inject_W[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint batch_size;
    uint seq_len;
    uint hidden_dim;
    uint capsule_dim;
    float injection_strength;  // Typically 0.1
    uint injection_mode;       // 0 = all positions, 1 = last position only, 2 = first position only
};

void main() {
    uint gID = gl_GlobalInvocationID.x;
    
    // Each thread handles one element of hidden states
    uint total_elements = batch_size * seq_len * hidden_dim;
    if (gID >= total_elements) return;
    
    // Decode position
    uint batch_idx = gID / (seq_len * hidden_dim);
    uint remainder = gID % (seq_len * hidden_dim);
    uint seq_idx = remainder / hidden_dim;
    uint hidden_idx = remainder % hidden_dim;
    
    // Check injection mode
    bool should_inject = false;
    if (injection_mode == 0) {
        // All positions
        should_inject = true;
    } else if (injection_mode == 1) {
        // Last position only (for generation)
        should_inject = (seq_idx == seq_len - 1);
    } else if (injection_mode == 2) {
        // First position only (CLS token style)
        should_inject = (seq_idx == 0);
    }
    
    if (!should_inject) return;
    
    // Project capsule → hidden: injection[hidden_idx] = sum(capsule[c] * W[hidden_idx, c])
    float injection = 0.0;
    for (uint c = 0; c < capsule_dim; c++) {
        uint capsule_idx = batch_idx * capsule_dim + c;
        uint w_idx = hidden_idx * capsule_dim + c;
        injection += memory_capsules[capsule_idx] * inject_W[w_idx];
    }
    
    // Scale and add to hidden state
    hidden[gID] += injection * injection_strength;
}
