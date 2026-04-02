#version 450

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Input vector (batch_size, in_features)
layout(set = 0, binding = 0) readonly buffer Input {
    float input_data[];
};

// Weight patterns / templates (out_features, in_features)
layout(set = 0, binding = 1) readonly buffer Weights {
    float weight_patterns[];
};

// Bias (out_features) — optional, can be zeros
layout(set = 0, binding = 2) readonly buffer Bias {
    float bias[];
};

// Output (batch_size, out_features)
layout(set = 0, binding = 3) buffer Output {
    float output_data[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint batch_size;
    uint in_features;
    uint out_features;
    uint use_bias;     // 0 or 1
};

// Addition-only linear: y[b][o] = -sum_i |weight[o][i] - input[b][i]| + bias[o]
// No multiplications — only additions, subtractions, absolute values.
// This is a radial basis function using Manhattan (L1) distance.
// The closer the input is to weight row o, the higher (less negative) the output.
void main() {
    uint gID = gl_GlobalInvocationID.x;

    // Each thread handles one (batch, output) pair
    uint total = batch_size * out_features;
    if (gID >= total) {
        return;
    }

    uint b = gID / out_features;  // batch index
    uint o = gID % out_features;  // output neuron index

    // Compute L1 distance: sum_i |w[o][i] - x[b][i]|
    float dist = 0.0;
    uint w_offset = o * in_features;
    uint x_offset = b * in_features;

    for (uint i = 0; i < in_features; i++) {
        dist += abs(weight_patterns[w_offset + i] - input_data[x_offset + i]);
    }

    // Output = -distance + bias (closer → higher value)
    float result = -dist;
    if (use_bias == 1) {
        result += bias[o];
    }

    output_data[gID] = result;
}
