#version 450

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Current agent position (2D or 3D)
layout(set = 0, binding = 0) readonly buffer AgentPosition {
    float agent_pos[]; // [x, y] or [x, y, z]
};

// Place field centers for each neuron
layout(set = 0, binding = 1) readonly buffer PlaceFieldCenters {
    float field_centers[]; // [x1, y1, x2, y2, ...] or [x1, y1, z1, x2, y2, z2, ...]
};

// Output firing rates
layout(set = 0, binding = 2) buffer FiringRates {
    float rates[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint n_neurons;           // Number of place cells
    uint spatial_dims;        // 2 for 2D, 3 for 3D space
    float field_width;        // Width of place field (standard deviation)
    float max_rate;           // Maximum firing rate (e.g., 20 Hz)
    float baseline_rate;      // Baseline firing rate (e.g., 0.1 Hz)
};

void main() {
    uint gID = gl_GlobalInvocationID.x;
    
    if (gID >= n_neurons) {
        return;
    }
    
    // Get place field center for this neuron
    uint center_offset = gID * spatial_dims;
    
    // Calculate squared distance from agent to place field center
    float dist_sq = 0.0;
    
    if (spatial_dims == 2) {
        // 2D spatial encoding
        float dx = agent_pos[0] - field_centers[center_offset];
        float dy = agent_pos[1] - field_centers[center_offset + 1];
        dist_sq = dx * dx + dy * dy;
    } else if (spatial_dims == 3) {
        // 3D spatial encoding
        float dx = agent_pos[0] - field_centers[center_offset];
        float dy = agent_pos[1] - field_centers[center_offset + 1];
        float dz = agent_pos[2] - field_centers[center_offset + 2];
        dist_sq = dx * dx + dy * dy + dz * dz;
    }
    
    // Gaussian tuning curve: rate = baseline + (max - baseline) * exp(-dist^2 / (2 * width^2))
    float sigma_sq = field_width * field_width;
    float gaussian = exp(-dist_sq / (2.0 * sigma_sq));
    
    // Calculate firing rate
    float rate = baseline_rate + (max_rate - baseline_rate) * gaussian;
    
    // Store output
    rates[gID] = rate;
}
