#version 450

layout (local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

// Output weight matrix (output_dim, input_dim)
layout(set = 0, binding = 0) buffer Weights {
    float weights[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint input_dim;
    uint output_dim;
    float scale;        // sqrt(2.0 / input_dim) for Xavier initialization
    uint seed;          // Random seed for reproducibility
};

// Hash function for pseudo-random number generation
// Based on PCG hash: https://www.pcg-random.org/
uint hash(uint x) {
    x ^= x >> 16;
    x *= 0x85ebca6b;
    x ^= x >> 13;
    x *= 0xc2b2ae35;
    x ^= x >> 16;
    return x;
}

// Generate uniform random number in [0, 1]
float random(uint x, uint y, uint seed) {
    uint h = hash(x + hash(y + hash(seed)));
    return float(h) / 4294967296.0;  // Divide by 2^32
}

// Box-Muller transform to generate normal distribution from uniform
// Returns two independent normal random numbers
vec2 box_muller(float u1, float u2) {
    float r = sqrt(-2.0 * log(max(u1, 1e-10)));
    float theta = 2.0 * 3.14159265359 * u2;
    return vec2(r * cos(theta), r * sin(theta));
}

void main() {
    // Each thread computes one weight: weights[output_idx][input_idx]
    uint output_idx = gl_GlobalInvocationID.y;  // Row (output dimension)
    uint input_idx = gl_GlobalInvocationID.x;  // Column (input dimension)
    
    if (output_idx >= output_dim || input_idx >= input_dim) {
        return;
    }
    
    // Generate two uniform random numbers
    float u1 = random(output_idx, input_idx, seed);
    float u2 = random(output_idx, input_idx + 1, seed);
    
    // Convert to normal distribution using Box-Muller transform
    vec2 normal = box_muller(u1, u2);
    
    // Use first normal value, scale by Xavier factor
    // Weight index: row-major order (output_dim, input_dim)
    uint weight_idx = output_idx * input_dim + input_idx;
    weights[weight_idx] = normal.x * scale;
}
