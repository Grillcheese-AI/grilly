#version 450

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Input tensor
layout(set = 0, binding = 0) readonly buffer Input {
    float input_data[];
};

// Random mask (same shape as input, values in [0, 1])
layout(set = 0, binding = 1) readonly buffer RandomMask {
    float random[];
};

// Output tensor
layout(set = 0, binding = 2) buffer Output {
    float output_data[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint total_elements;
    float dropout_prob;   // Probability of dropping (e.g., 0.1)
    uint is_training;     // 1 = training mode, 0 = inference mode
};

void main() {
    uint gID = gl_GlobalInvocationID.x;
    
    if (gID >= total_elements) {
        return;
    }
    
    float x = input_data[gID];
    
    if (is_training == 1) {
        // Training mode: apply dropout with scaling
        float rand_val = random[gID];
        float keep_prob = 1.0 - dropout_prob;
        
        // If random value > dropout_prob, keep the value and scale
        // Otherwise, zero it out
        float mask = (rand_val >= dropout_prob) ? 1.0 : 0.0;
        output_data[gID] = x * mask / keep_prob;  // Inverted dropout
    } else {
        // Inference mode: pass through
        output_data[gID] = x;
    }
}
