#version 450
#extension GL_EXT_shader_atomic_float : enable

/*
 * AvgPool2d Backward Pass
 *
 * Distributes gradients evenly to all positions in pooling window.
 */

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(binding = 0) readonly buffer GradOutputBuffer {
    float grad_output[];
};

layout(binding = 1) buffer GradInputBuffer {
    float grad_input[];
};

layout(push_constant) uniform PushConstants {
    uint batch_size;
    uint channels;
    uint in_height;
    uint in_width;
    uint out_height;
    uint out_width;
    uint kernel_h;
    uint kernel_w;
    uint stride_h;
    uint stride_w;
    uint padding_h;
    uint padding_w;
    uint count_include_pad;
} params;

void main() {
    // Calculate output position
    uint batch = gl_GlobalInvocationID.z / params.channels;
    uint channel = gl_GlobalInvocationID.z % params.channels;
    uint out_y = gl_GlobalInvocationID.y;
    uint out_x = gl_GlobalInvocationID.x;

    if (batch >= params.batch_size || out_y >= params.out_height || out_x >= params.out_width) {
        return;
    }

    // Get gradient from output
    uint grad_out_idx = batch * params.channels * params.out_height * params.out_width +
                       channel * params.out_height * params.out_width +
                       out_y * params.out_width +
                       out_x;
    float grad = grad_output[grad_out_idx];

    // Count valid positions in window
    uint count = 0;
    for (uint kh = 0; kh < params.kernel_h; kh++) {
        for (uint kw = 0; kw < params.kernel_w; kw++) {
            int in_y = int(out_y * params.stride_h + kh) - int(params.padding_h);
            int in_x = int(out_x * params.stride_w + kw) - int(params.padding_w);

            if (in_y >= 0 && in_y < int(params.in_height) &&
                in_x >= 0 && in_x < int(params.in_width)) {
                count++;
            } else if (params.count_include_pad == 1) {
                count++;
            }
        }
    }

    float grad_per_element = (count > 0) ? (grad / float(count)) : 0.0;

    // Distribute gradient to input positions
    for (uint kh = 0; kh < params.kernel_h; kh++) {
        for (uint kw = 0; kw < params.kernel_w; kw++) {
            int in_y = int(out_y * params.stride_h + kh) - int(params.padding_h);
            int in_x = int(out_x * params.stride_w + kw) - int(params.padding_w);

            if (in_y >= 0 && in_y < int(params.in_height) &&
                in_x >= 0 && in_x < int(params.in_width)) {

                uint grad_in_idx = batch * params.channels * params.in_height * params.in_width +
                                  channel * params.in_height * params.in_width +
                                  uint(in_y) * params.in_width +
                                  uint(in_x);

                atomicAdd(grad_input[grad_in_idx], grad_per_element);
            }
        }
    }
}
