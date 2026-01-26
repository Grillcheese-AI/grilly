#version 450

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Logits (batch, seq_len, vocab_size)
layout(set = 0, binding = 0) readonly buffer Logits {
    float logits[];
};

// Predicted token IDs (batch, seq_len)
layout(set = 0, binding = 1) buffer PredictedTokens {
    uint predictions[];
};

// Prediction scores (batch, seq_len) - optional
layout(set = 0, binding = 2) buffer PredictionScores {
    float scores[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint batch_size;
    uint seq_len;
    uint vocab_size;
    uint output_scores;   // 1 = also output max scores, 0 = tokens only
};

void main() {
    uint gID = gl_GlobalInvocationID.x;
    uint total_positions = batch_size * seq_len;
    
    if (gID >= total_positions) {
        return;
    }
    
    uint batch_idx = gID / seq_len;
    uint seq_idx = gID % seq_len;
    
    // Find argmax
    float max_val = -1e10;
    uint max_idx = 0;
    
    for (uint v = 0; v < vocab_size; v++) {
        uint logit_idx = batch_idx * seq_len * vocab_size + 
                        seq_idx * vocab_size + v;
        float val = logits[logit_idx];
        
        if (val > max_val) {
            max_val = val;
            max_idx = v;
        }
    }
    
    predictions[gID] = max_idx;
    
    if (output_scores == 1) {
        scores[gID] = max_val;
    }
}
