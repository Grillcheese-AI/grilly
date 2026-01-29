#version 450

/*
 * 2D Convolution Backward Pass - Gradient w.r.t. Input
 *
 * Computes gradient of loss w.r.t. input using transposed convolution.
 * This is the backward pass for the input tensor.
 *
 * grad_output shape: (batch, out_channels, out_h, out_w)
 * weight shape: (out_channels, in_channels/groups, kernel_h, kernel_w)
 * grad_input shape: (batch, in_channels, in_h, in_w)
 */

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(binding = 0) readonly buffer GradOutputBuffer {
    float grad_output[];
};

layout(binding = 1) readonly buffer WeightBuffer {
    float weight_data[];
};

layout(binding = 2) buffer GradInputBuffer {
    float grad_input[];  // Using buffer (not writeonly) to allow atomic adds
};

layout(push_constant) uniform PushConstants {
    uint batch_size;
    uint in_channels;
    uint in_height;
    uint in_width;
    uint out_channels;
    uint out_height;
    uint out_width;
    uint kernel_h;
    uint kernel_w;
    uint stride_h;
    uint stride_w;
    uint padding_h;
    uint padding_w;
    uint dilation_h;
    uint dilation_w;
    uint groups;
} params;

void main() {
    // Calculate input position (includes channel dimension)
    uint batch = gl_GlobalInvocationID.z / params.in_channels;
    uint ic = gl_GlobalInvocationID.z % params.in_channels;
    uint in_y = gl_GlobalInvocationID.y;
    uint in_x = gl_GlobalInvocationID.x;

    // Bounds check
    if (batch >= params.batch_size || ic >= params.in_channels ||
        in_y >= params.in_height   || in_x >= params.in_width) {
        return;
    }

    uint in_channels_per_group = params.in_channels / params.groups;
    uint out_channels_per_group = params.out_channels / params.groups;

    // Process this input channel
    float grad_sum = 0.0;

    // Determine group for this input channel
    uint group = ic / in_channels_per_group;
    uint out_channel_start = group * out_channels_per_group;
    uint out_channel_end = out_channel_start + out_channels_per_group;

        // Iterate over output channels in this group
        for (uint oc = out_channel_start; oc < out_channel_end; oc++) {
            // Iterate over kernel positions
            for (uint kh = 0; kh < params.kernel_h; kh++) {
                for (uint kw = 0; kw < params.kernel_w; kw++) {
                    // Calculate which output position this input affects
                    // out_y * stride + kh * dilation - padding = in_y
                    // Solve for out_y: out_y = (in_y + padding - kh * dilation) / stride

                    int numerator_y = int(in_y) + int(params.padding_h) - int(kh * params.dilation_h);
                    int numerator_x = int(in_x) + int(params.padding_w) - int(kw * params.dilation_w);

                    // Check if divisible by stride (otherwise this kernel position doesn't contribute)
                    if (numerator_y % int(params.stride_h) == 0 && numerator_x % int(params.stride_w) == 0) {
                        int out_y = numerator_y / int(params.stride_h);
                        int out_x = numerator_x / int(params.stride_w);

                        // Check if in valid output range
                        if (out_y >= 0 && out_y < int(params.out_height) &&
                            out_x >= 0 && out_x < int(params.out_width)) {

                            // Calculate indices
                            uint grad_out_idx = batch * params.out_channels * params.out_height * params.out_width +
                                               oc * params.out_height * params.out_width +
                                               uint(out_y) * params.out_width +
                                               uint(out_x);

                            uint weight_idx = oc * in_channels_per_group * params.kernel_h * params.kernel_w +
                                             (ic - group * in_channels_per_group) * params.kernel_h * params.kernel_w +
                                             kh * params.kernel_w +
                                             kw;

                            grad_sum += grad_output[grad_out_idx] * weight_data[weight_idx];
                        }
                    }
                }
            }
    }

    // Write gradient
    uint grad_input_idx = batch * params.in_channels * params.in_height * params.in_width +
                         ic * params.in_height * params.in_width +
                         in_y * params.in_width +
                         in_x;
    grad_input[grad_input_idx] = grad_sum;
}
