#version 450

/*
 * MaxPool2d Backward Pass
 *
 * Dispatches over INPUT positions (avoids race conditions, no atomics needed).
 * Uses indices from forward pass to route gradients.
 */

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(binding = 0) readonly buffer GradOutputBuffer {
    float grad_output[];
};

layout(binding = 1) readonly buffer IndicesBuffer {
    uint indices[];  // Max indices from forward pass
};

layout(binding = 2) writeonly buffer GradInputBuffer {
    float grad_input[];
};

layout(push_constant) uniform PushConstants {
    uint batch_size;
    uint channels;
    uint in_height;
    uint in_width;
    uint out_height;
    uint out_width;
} params;

void main() {
    // Calculate INPUT position (dispatch over input space)
    uint batch = gl_GlobalInvocationID.z / params.channels;
    uint channel = gl_GlobalInvocationID.z % params.channels;
    uint in_y = gl_GlobalInvocationID.y;
    uint in_x = gl_GlobalInvocationID.x;

    if (batch >= params.batch_size || in_y >= params.in_height || in_x >= params.in_width) {
        return;
    }

    // Calculate this input's linear index
    uint in_idx = batch * params.channels * params.in_height * params.in_width +
                  channel * params.in_height * params.in_width +
                  in_y * params.in_width +
                  in_x;

    float grad_sum = 0.0;

    // Check all output positions to see if they selected this input as max
    uint output_size = params.out_height * params.out_width;
    uint output_offset = batch * params.channels * output_size + channel * output_size;

    for (uint out_idx = 0; out_idx < output_size; out_idx++) {
        uint global_out_idx = output_offset + out_idx;

        // Check if this output position's max came from our input position
        if (indices[global_out_idx] == in_idx) {
            grad_sum += grad_output[global_out_idx];
        }
    }

    // Write gradient
    grad_input[in_idx] = grad_sum;
}
