#version 450

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Hidden states (batch, seq_len, dim)
layout(set = 0, binding = 0) readonly buffer HiddenStates {
    float hidden[];
};

// Memory context (batch, dim) - weighted sum of retrieved memories
layout(set = 0, binding = 1) readonly buffer MemoryContext {
    float context[];
};

// Augmented output (batch, seq_len, dim)
layout(set = 0, binding = 2) buffer Output {
    float output_data[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint batch_size;
    uint seq_len;
    uint dim;
    float alpha;         // Blending strength (e.g., 0.1)
};

void main() {
    uint gID = gl_GlobalInvocationID.x;
    uint total_elements = batch_size * seq_len * dim;
    
    if (gID >= total_elements) {
        return;
    }
    
    uint batch_idx = gID / (seq_len * dim);
    uint remainder = gID % (seq_len * dim);
    uint dim_idx = remainder % dim;
    
    // Get hidden state
    float h = hidden[gID];
    
    // Get memory context (broadcast over sequence dimension)
    uint context_idx = batch_idx * dim + dim_idx;
    float mem = context[context_idx];
    
    // Simple additive injection: output = hidden + alpha * memory
    output_data[gID] = h + alpha * mem;
}
