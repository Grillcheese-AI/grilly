#version 450
#extension GL_EXT_shader_atomic_float : require

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Errors from batch (batch_size)
layout(set = 0, binding = 0) readonly buffer Errors {
    float errors[];
};

// Accumulated metrics
layout(set = 0, binding = 1) buffer Metrics {
    float total_error_sq[];
    uint update_count[];
    float last_error[];
};

// Output RMSE
layout(set = 0, binding = 2) buffer RMSE {
    float rmse[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint batch_size;
    uint pass_type;      // 0 = accumulate, 1 = compute RMSE
};

void main() {
    uint gID = gl_GlobalInvocationID.x;
    
    if (pass_type == 0) {
        // Pass 1: Accumulate error statistics
        if (gID >= batch_size) return;
        
        float err = errors[gID];
        
        // Atomic add to total squared error
        atomicAdd(total_error_sq[0], err * err);
        atomicAdd(update_count[0], 1);
        
        // Store last error (not atomic, but gives approximate value)
        last_error[0] = err;
        
    } else if (pass_type == 1) {
        // Pass 2: Compute RMSE
        if (gID > 0) return;
        
        uint count = update_count[0];
        if (count > 0) {
            rmse[0] = sqrt(total_error_sq[0] / float(count));
        } else {
            rmse[0] = 1e10;  // Infinity approximation
        }
    }
}
