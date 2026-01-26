#version 450

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Input data (batch, seq, features)
layout(set = 0, binding = 0) readonly buffer Input {
    float input_data[];
};

// Output expanded in time dimension (batch, seq, timesteps, features)
layout(set = 0, binding = 1) buffer OutputExpanded {
    float output_data[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint batch_size;
    uint seq_len;
    uint num_timesteps;
    uint features;
};

void main() {
    uint gID = gl_GlobalInvocationID.x;
    uint total_input_elements = batch_size * seq_len * features;
    
    if (gID >= total_input_elements) {
        return;
    }
    
    // Decode input position: [batch, seq, feat]
    uint batch_idx = gID / (seq_len * features);
    uint remainder = gID % (seq_len * features);
    uint seq_idx = remainder / features;
    uint feat_idx = remainder % features;
    
    // Read input value
    float val = input_data[gID];
    
    // Replicate across timesteps in output
    // Output layout: [batch, seq, time, feat]
    for (uint t = 0; t < num_timesteps; t++) {
        uint out_idx = batch_idx * (seq_len * num_timesteps * features) +
                       seq_idx * (num_timesteps * features) +
                       t * features +
                       feat_idx;
        output_data[out_idx] = val;
    }
}
