#version 450

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Logits (batch, seq_len, vocab_size)
layout(set = 0, binding = 0) readonly buffer Logits {
    float logits[];
};

// Target token IDs (batch, seq_len)
layout(set = 0, binding = 1) readonly buffer Targets {
    uint target_ids[];
};

// Per-token losses (batch, seq_len)
layout(set = 0, binding = 2) buffer Losses {
    float losses[];
};

// Max logits buffer (batch, seq_len) - for numerical stability
layout(set = 0, binding = 3) buffer MaxLogits {
    float max_logits[];
};

// Sum exp buffer (batch, seq_len)
layout(set = 0, binding = 4) buffer SumExp {
    float sum_exp[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint batch_size;
    uint seq_len;
    uint vocab_size;
    uint pass_type;      // 0 = max, 1 = sum_exp, 2 = loss
    float label_smoothing; // Optional (e.g., 0.1)
};

void main() {
    uint gID = gl_GlobalInvocationID.x;
    
    if (pass_type == 0) {
        // Pass 1: Compute max logit per position
        uint total_positions = batch_size * seq_len;
        if (gID >= total_positions) return;
        
        uint batch_idx = gID / seq_len;
        uint seq_idx = gID % seq_len;
        
        float max_val = -1e10;
        for (uint v = 0; v < vocab_size; v++) {
            uint logit_idx = batch_idx * seq_len * vocab_size + 
                           seq_idx * vocab_size + v;
            max_val = max(max_val, logits[logit_idx]);
        }
        max_logits[gID] = max_val;
        
    } else if (pass_type == 1) {
        // Pass 2: Compute sum of exp(logit - max)
        uint total_positions = batch_size * seq_len;
        if (gID >= total_positions) return;
        
        uint batch_idx = gID / seq_len;
        uint seq_idx = gID % seq_len;
        float max_val = max_logits[gID];
        
        float sum = 0.0;
        for (uint v = 0; v < vocab_size; v++) {
            uint logit_idx = batch_idx * seq_len * vocab_size + 
                           seq_idx * vocab_size + v;
            sum += exp(logits[logit_idx] - max_val);
        }
        sum_exp[gID] = sum;
        
    } else if (pass_type == 2) {
        // Pass 3: Compute cross-entropy loss
        uint total_positions = batch_size * seq_len;
        if (gID >= total_positions) return;
        
        uint batch_idx = gID / seq_len;
        uint seq_idx = gID % seq_len;
        uint target_id = target_ids[gID];
        
        // Bounds check
        if (target_id >= vocab_size) {
            losses[gID] = 0.0;
            return;
        }
        
        float max_val = max_logits[gID];
        float sum = sum_exp[gID];
        
        // Get logit for target class
        uint target_logit_idx = batch_idx * seq_len * vocab_size + 
                               seq_idx * vocab_size + target_id;
        float target_logit = logits[target_logit_idx];
        
        // Cross-entropy: -log(softmax(target))
        // = -(target_logit - max) + log(sum_exp)
        float log_prob = (target_logit - max_val) - log(sum);
        float loss = -log_prob;
        
        // Optional label smoothing
        if (label_smoothing > 0.0) {
            float smooth_loss = label_smoothing * (-log(1.0 / float(vocab_size)));
            loss = (1.0 - label_smoothing) * loss + smooth_loss;
        }
        
        losses[gID] = loss;
    }
}
