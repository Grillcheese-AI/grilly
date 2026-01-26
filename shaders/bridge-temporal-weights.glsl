#version 450

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Output temporal weights (num_timesteps)
layout(set = 0, binding = 0) buffer TemporalWeights {
    float weights[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint num_timesteps;
    float time_window;    // Decay window parameter
};

void main() {
    uint t = gl_GlobalInvocationID.x;
    
    if (t >= num_timesteps) {
        return;
    }
    
    // Exponential weighting: exp(t / time_window)
    // More recent timesteps get higher weight
    float weight = exp(float(t) / time_window);
    weights[t] = weight;
}
