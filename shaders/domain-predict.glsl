#version 450

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Domain logits (batch, num_domains)
layout(set = 0, binding = 0) readonly buffer DomainLogits {
    float logits[];
};

// Domain predictions (batch) - argmax
layout(set = 0, binding = 1) buffer DomainPredictions {
    uint predictions[];
};

// Confidence scores (batch)
layout(set = 0, binding = 2) buffer ConfidenceScores {
    float confidence[];
};

// Entropy scores (batch) - uncertainty measure
layout(set = 0, binding = 3) buffer EntropyScores {
    float entropy[];
};

// Softmax probabilities (batch, num_domains)
layout(set = 0, binding = 4) buffer Probabilities {
    float probs[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint batch_size;
    uint num_domains;
    uint pass_type;      // 0 = softmax, 1 = argmax + confidence
};

void main() {
    uint batch_idx = gl_GlobalInvocationID.x;
    
    if (batch_idx >= batch_size) {
        return;
    }
    
    if (pass_type == 0) {
        // Pass 1: Compute softmax
        float max_logit = -1e10;
        for (uint d = 0; d < num_domains; d++) {
            uint idx = batch_idx * num_domains + d;
            max_logit = max(max_logit, logits[idx]);
        }
        
        float sum_exp = 0.0;
        for (uint d = 0; d < num_domains; d++) {
            uint idx = batch_idx * num_domains + d;
            float exp_val = exp(logits[idx] - max_logit);
            probs[idx] = exp_val;
            sum_exp += exp_val;
        }
        
        for (uint d = 0; d < num_domains; d++) {
            uint idx = batch_idx * num_domains + d;
            probs[idx] /= sum_exp;
        }
        
    } else if (pass_type == 1) {
        // Pass 2: Argmax and confidence metrics
        float max_prob = -1e10;
        uint argmax_domain = 0;
        float ent = 0.0;
        
        for (uint d = 0; d < num_domains; d++) {
            uint idx = batch_idx * num_domains + d;
            float p = probs[idx];
            
            if (p > max_prob) {
                max_prob = p;
                argmax_domain = d;
            }
            
            // Entropy: -sum(p * log(p))
            if (p > 1e-10) {
                ent -= p * log(p);
            }
        }
        
        predictions[batch_idx] = argmax_domain;
        confidence[batch_idx] = max_prob;
        entropy[batch_idx] = ent;
    }
}
