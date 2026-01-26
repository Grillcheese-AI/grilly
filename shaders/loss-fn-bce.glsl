#version 450
#extension GL_KHR_shader_subgroup_arithmetic : enable // Optional: for more efficient reduction

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in; // Define workgroup size

// Input buffers (predicted probabilities and true labels)
layout(set = 0, binding = 0) readonly buffer Probabilities {
    float p_pred[];
};

layout(set = 0, binding = 1) readonly buffer Labels {
    float y_true[];
};

// Output buffer for individual loss contributions
layout(set = 0, binding = 2) buffer ElementLosses {
    float losses[];
};

// Push constant for the total number of elements
layout(push_constant) uniform PushConsts {
    uint n_elements;
};

void main() {
    uint gID = gl_GlobalInvocationID.x; // Get a unique global thread ID

    // Ensure we don't access memory outside the buffer bounds
    if (gID >= n_elements) {
        return;
    }

    float p = p_pred[gID];
    float y = y_true[gID];

    // Ensure numerical stability by clamping probabilities to a small range (e.g., [1e-7, 1.0 - 1e-7])
    // The log of 0 is undefined (approaches -infinity), which causes issues.
    float epsilon = 1e-7;
    p = clamp(p, epsilon, 1.0 - epsilon);

    // Calculate binary cross-entropy for a single element:
    // Loss = - (y * log(p) + (1 - y) * log(1 - p))
    float bce_loss = - (y * log(p) + (1.0 - y) * log(1.0 - p));

    // Store the result in the output buffer
    losses[gID] = bce_loss;
}
