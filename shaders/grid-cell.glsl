#version 450

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Current agent position (2D)
layout(set = 0, binding = 0) readonly buffer AgentPosition {
    float agent_pos[]; // [x, y]
};

// Grid cell parameters (per neuron)
layout(set = 0, binding = 1) readonly buffer GridSpacings {
    float spacings[]; // [s0, s1, ...]
};

layout(set = 0, binding = 2) readonly buffer GridOrientations {
    float orientations[]; // [theta0, theta1, ...]
};

layout(set = 0, binding = 3) readonly buffer GridPhases {
    float phases[]; // [px0, py0, px1, py1, ...]
};

// Output firing rates
layout(set = 0, binding = 4) buffer FiringRates {
    float rates[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint n_neurons;
    float max_rate;    // e.g. 25 Hz
};

// Hexagonal 3-wave grid cell activation
// Three cosine waves at 60° intervals create hexagonal firing pattern
// r = max_rate * relu((cos(u1) + cos(u2) + cos(u3)) / 3 + 0.5)
void main() {
    uint gID = gl_GlobalInvocationID.x;

    if (gID >= n_neurons) {
        return;
    }

    float x = agent_pos[0];
    float y = agent_pos[1];

    // Rotation by grid orientation
    float theta = orientations[gID];
    float cos_t = cos(theta);
    float sin_t = sin(theta);
    float rx = cos_t * x - sin_t * y;
    float ry = sin_t * x + cos_t * y;

    // Shift by phase
    uint phase_off = gID * 2;
    float sx = rx - phases[phase_off];
    float sy = ry - phases[phase_off + 1];

    // Wave vector: k = 4*pi / (sqrt(3) * spacing)
    float k = 4.0 * 3.14159265 / (1.7320508 * spacings[gID]);

    // Three waves at 60° intervals (hexagonal pattern)
    float u1 = k * sx;
    float u2 = k * (-0.5 * sx + 0.866025 * sy);
    float u3 = k * (-0.5 * sx - 0.866025 * sy);

    float val = (cos(u1) + cos(u2) + cos(u3)) / 3.0 + 0.5;
    rates[gID] = max_rate * max(val, 0.0);
}
