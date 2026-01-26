#version 450

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Hidden states (batch, seq_len, dim)
layout(set = 0, binding = 0) readonly buffer HiddenStates {
    float hidden[];
};

// Memory context (batch, dim)
layout(set = 0, binding = 1) readonly buffer MemoryContext {
    float context[];
};

// Gate weights (dim, dim * 2)
layout(set = 0, binding = 2) readonly buffer GateWeights {
    float W_gate[];
};

// Gate bias (dim)
layout(set = 0, binding = 3) readonly buffer GateBias {
    float b_gate[];
};

// Memory projection weights (dim, dim)
layout(set = 0, binding = 4) readonly buffer MemoryProjWeights {
    float W_proj[];
};

// Augmented output (batch, seq_len, dim)
layout(set = 0, binding = 5) buffer Output {
    float output_data[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint batch_size;
    uint seq_len;
    uint dim;
};

// Sigmoid activation
float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

void main() {
    uint gID = gl_GlobalInvocationID.x;
    uint total_elements = batch_size * seq_len * dim;
    
    if (gID >= total_elements) {
        return;
    }
    
    uint batch_idx = gID / (seq_len * dim);
    uint remainder = gID % (seq_len * dim);
    uint seq_idx = remainder / dim;
    uint dim_idx = remainder % dim;
    
    // Get hidden state
    float h = hidden[gID];
    
    // Get memory context (broadcast over sequence)
    uint context_idx = batch_idx * dim + dim_idx;
    float mem_raw = context[context_idx];
    
    // Project memory
    float mem_proj = 0.0;
    for (uint k = 0; k < dim; k++) {
        uint ctx_idx = batch_idx * dim + k;
        uint w_idx = dim_idx * dim + k;
        mem_proj += W_proj[w_idx] * context[ctx_idx];
    }
    
    // Compute gate value
    // Concatenate hidden and memory, then apply linear + sigmoid
    float gate_val = 0.0;
    
    // Contribution from hidden state (first half of concatenation)
    for (uint k = 0; k < dim; k++) {
        uint h_idx = batch_idx * seq_len * dim + seq_idx * dim + k;
        uint w_idx = dim_idx * (2 * dim) + k;
        gate_val += W_gate[w_idx] * hidden[h_idx];
    }
    
    // Contribution from memory context (second half of concatenation)
    for (uint k = 0; k < dim; k++) {
        uint ctx_idx = batch_idx * dim + k;
        uint w_idx = dim_idx * (2 * dim) + dim + k;
        gate_val += W_gate[w_idx] * context[ctx_idx];
    }
    
    gate_val += b_gate[dim_idx];
    float gate = sigmoid(gate_val);
    
    // Gated injection: output = hidden + gate * memory_proj
    output_data[gID] = h + gate * mem_proj;
}
