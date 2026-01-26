#version 450

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Softmax probabilities (batch, seq_len, vocab_size)
layout(set = 0, binding = 0) readonly buffer Probabilities {
    float probs[];
};

// Random values (batch, seq_len) in [0, 1]
layout(set = 0, binding = 1) readonly buffer RandomValues {
    float randoms[];
};

// Sampled token IDs (batch, seq_len)
layout(set = 0, binding = 2) buffer SampledTokens {
    uint samples[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint batch_size;
    uint seq_len;
    uint vocab_size;
};

void main() {
    uint gID = gl_GlobalInvocationID.x;
    uint total_positions = batch_size * seq_len;
    
    if (gID >= total_positions) {
        return;
    }
    
    uint batch_idx = gID / seq_len;
    uint seq_idx = gID % seq_len;
    float random_val = randoms[gID];
    
    // Categorical sampling using inverse CDF
    float cumsum = 0.0;
    uint sampled_token = 0;
    
    for (uint v = 0; v < vocab_size; v++) {
        uint prob_idx = batch_idx * seq_len * vocab_size + 
                       seq_idx * vocab_size + v;
        cumsum += probs[prob_idx];
        
        if (random_val <= cumsum) {
            sampled_token = v;
            break;
        }
    }
    
    samples[gID] = sampled_token;
}
