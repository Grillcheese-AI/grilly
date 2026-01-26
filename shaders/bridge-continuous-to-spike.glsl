#version 450

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Input continuous features (batch, input_dim)
layout(set = 0, binding = 0) readonly buffer InputFeatures {
    float features[];
};

// Output spikes (batch, time, spike_dim)
layout(set = 0, binding = 1) buffer OutputSpikes {
    float spikes[];
};

// Projection matrix (spike_dim, input_dim) if needed
layout(set = 0, binding = 2) readonly buffer ProjectionMatrix {
    float W[];
};

// Projection bias (spike_dim) if needed
layout(set = 0, binding = 3) readonly buffer ProjectionBias {
    float b[];
};

// Random numbers for Poisson encoding (batch, time, spike_dim)
layout(set = 0, binding = 4) readonly buffer RandomNumbers {
    float random[];
};

// Temporary projected features (batch, spike_dim)
layout(set = 0, binding = 5) buffer TempProjected {
    float temp[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint batch_size;
    uint num_timesteps;
    uint input_dim;
    uint spike_dim;
    uint encoding_type;      // 0=poisson, 1=temporal
    uint use_projection;     // 1 if projection needed, 0 for identity
    uint pass_type;          // 0=project features, 1=encode to spikes
};

// Sigmoid activation
float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

void main() {
    uint gID = gl_GlobalInvocationID.x;
    
    if (pass_type == 0) {
        // Pass 1: Project features (if needed)
        uint total_projected = batch_size * spike_dim;
        if (gID >= total_projected) return;
        
        uint batch_idx = gID / spike_dim;
        uint spike_idx = gID % spike_dim;
        
        if (use_projection == 0) {
            // Identity: just copy
            uint feat_idx = batch_idx * input_dim + spike_idx;
            temp[gID] = features[feat_idx];
        } else {
            // Linear projection
            float sum = 0.0;
            for (uint k = 0; k < input_dim; k++) {
                uint feat_idx = batch_idx * input_dim + k;
                uint w_idx = spike_idx * input_dim + k;
                sum += features[feat_idx] * W[w_idx];
            }
            sum += b[spike_idx];
            temp[gID] = sum;
        }
        
    } else if (pass_type == 1) {
        // Pass 2: Encode to spikes
        uint total_spikes = batch_size * num_timesteps * spike_dim;
        if (gID >= total_spikes) return;
        
        uint batch_idx = gID / (num_timesteps * spike_dim);
        uint remainder = gID % (num_timesteps * spike_dim);
        uint time_idx = remainder / spike_dim;
        uint spike_idx = remainder % spike_dim;
        
        // Get projected feature
        uint temp_idx = batch_idx * spike_dim + spike_idx;
        float feat = temp[temp_idx];
        
        float spike_val = 0.0;
        
        if (encoding_type == 0) {
            // Poisson encoding: spike probability = sigmoid(feat)
            float rate = sigmoid(feat);
            float rand_val = random[gID];
            spike_val = (rand_val < rate) ? 1.0 : 0.0;
            
        } else if (encoding_type == 1) {
            // Temporal encoding: spike at time proportional to value
            // Higher values spike earlier
            float norm = sigmoid(feat) * float(num_timesteps);
            float time_f = float(time_idx);
            spike_val = (norm > time_f) ? 1.0 : 0.0;
        }
        
        spikes[gID] = spike_val;
    }
}
