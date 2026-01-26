#version 450

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Input spikes (batch, time, spike_dim)
layout(set = 0, binding = 0) readonly buffer InputSpikes {
    float spikes[];
};

// Output continuous features (batch, output_dim)
layout(set = 0, binding = 1) buffer OutputFeatures {
    float features[];
};

// Optional: temporal weights for temporal encoding
layout(set = 0, binding = 2) readonly buffer TemporalWeights {
    float weights[];
};

// Optional: projection matrix (output_dim, spike_dim)
layout(set = 0, binding = 3) readonly buffer ProjectionMatrix {
    float W[];
};

// Optional: projection bias (output_dim)
layout(set = 0, binding = 4) readonly buffer ProjectionBias {
    float b[];
};

// Temporary buffer for intermediate results (batch, spike_dim)
layout(set = 0, binding = 5) buffer TempBuffer {
    float temp[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint batch_size;
    uint total_time;
    uint spike_dim;
    uint output_dim;
    uint time_window;        // Window for averaging
    uint encoding_type;      // 0=rate, 1=temporal, 2=phase
    uint use_projection;     // 1 if projection needed, 0 for identity
    uint pass_type;          // 0=encode, 1=project (for 2-pass execution)
};

const float PI = 3.14159265359;

void main() {
    uint gID = gl_GlobalInvocationID.x;
    
    if (pass_type == 0) {
        // Pass 1: Encoding (spikes → intermediate features)
        uint total_batch_features = batch_size * spike_dim;
        if (gID >= total_batch_features) return;
        
        uint batch_idx = gID / spike_dim;
        uint feat_idx = gID % spike_dim;
        
        float result = 0.0;
        
        if (encoding_type == 0) {
            // Rate encoding: mean over recent time window
            uint start_t = total_time > time_window ? total_time - time_window : 0;
            float sum = 0.0;
            for (uint t = start_t; t < total_time; t++) {
                uint spike_idx = batch_idx * total_time * spike_dim + t * spike_dim + feat_idx;
                sum += spikes[spike_idx];
            }
            result = sum / float(time_window);
            
        } else if (encoding_type == 1) {
            // Temporal encoding: exponentially weighted average
            float weighted_sum = 0.0;
            float weight_sum = 0.0;
            for (uint t = 0; t < total_time; t++) {
                uint spike_idx = batch_idx * total_time * spike_dim + t * spike_dim + feat_idx;
                float w = weights[t];
                weighted_sum += spikes[spike_idx] * w;
                weight_sum += w;
            }
            result = weighted_sum / (weight_sum + 1e-6);
            
        } else if (encoding_type == 2) {
            // Phase encoding: FFT magnitude (simplified - use recent window mean for now)
            // Full FFT would require separate shader
            uint start_t = total_time > time_window ? total_time - time_window : 0;
            float sum = 0.0;
            for (uint t = start_t; t < total_time; t++) {
                uint spike_idx = batch_idx * total_time * spike_dim + t * spike_dim + feat_idx;
                sum += spikes[spike_idx];
            }
            result = sum / float(time_window);
        }
        
        temp[gID] = result;
        
    } else if (pass_type == 1) {
        // Pass 2: Projection (for when spike_dim != output_dim)
        uint total_outputs = batch_size * output_dim;
        if (gID >= total_outputs) return;
        
        uint batch_idx = gID / output_dim;
        uint out_idx_local = gID % output_dim;
        
        if (use_projection == 0) {
            // Identity mapping
            uint temp_idx = batch_idx * spike_dim + out_idx_local;
            features[gID] = temp[temp_idx];
        } else {
            // Linear projection: y = W @ x + b
            float sum = 0.0;
            for (uint k = 0; k < spike_dim; k++) {
                uint temp_idx = batch_idx * spike_dim + k;
                uint w_idx = out_idx_local * spike_dim + k;
                sum += temp[temp_idx] * W[w_idx];
            }
            sum += b[out_idx_local];
            features[gID] = sum;
        }
    }
}
