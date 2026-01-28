#version 450

/*
 * 2D Convolution Forward Pass
 *
 * Computes 2D convolution with support for:
 * - Stride, padding, dilation
 * - Grouped convolutions
 * - Bias addition
 *
 * Input shape: (batch, in_channels, height, width)
 * Weight shape: (out_channels, in_channels/groups, kernel_h, kernel_w)
 * Output shape: (batch, out_channels, out_h, out_w)
 *
 * Performance: Use workgroup size (8, 8, 1) for 2D tiling
 */

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(binding = 0) readonly buffer InputBuffer {
    float input_data[];
};

layout(binding = 1) readonly buffer WeightBuffer {
    float weight_data[];
};

layout(binding = 2) readonly buffer BiasBuffer {
    float bias_data[];  // Optional - size out_channels
};

layout(binding = 3) writeonly buffer OutputBuffer {
    float output_data[];
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
    uint has_bias;  // 1 if bias present, 0 otherwise
} params;

void main() {
    // Calculate output position
    uint batch = gl_GlobalInvocationID.z;
    uint out_y = gl_GlobalInvocationID.y;
    uint out_x = gl_GlobalInvocationID.x;

    // Bounds check
    if (batch >= params.batch_size || out_y >= params.out_height || out_x >= params.out_width) {
        return;
    }

    // Calculate channels per group
    uint in_channels_per_group = params.in_channels / params.groups;
    uint out_channels_per_group = params.out_channels / params.groups;

    // Process each output channel
    for (uint oc = 0; oc < params.out_channels; oc++) {
        float sum = 0.0;

        // Determine group for this output channel
        uint group = oc / out_channels_per_group;
        uint in_channel_start = group * in_channels_per_group;
        uint in_channel_end = in_channel_start + in_channels_per_group;

        // Convolve over kernel and input channels
        for (uint ic = in_channel_start; ic < in_channel_end; ic++) {
            for (uint kh = 0; kh < params.kernel_h; kh++) {
                for (uint kw = 0; kw < params.kernel_w; kw++) {
                    // Calculate input position with padding and dilation
                    int in_y = int(out_y * params.stride_h + kh * params.dilation_h) - int(params.padding_h);
                    int in_x = int(out_x * params.stride_w + kw * params.dilation_w) - int(params.padding_w);

                    // Check if in valid input range
                    if (in_y >= 0 && in_y < int(params.in_height) &&
                        in_x >= 0 && in_x < int(params.in_width)) {

                        // Calculate indices
                        uint input_idx = batch * params.in_channels * params.in_height * params.in_width +
                                        ic * params.in_height * params.in_width +
                                        uint(in_y) * params.in_width +
                                        uint(in_x);

                        uint weight_idx = oc * in_channels_per_group * params.kernel_h * params.kernel_w +
                                         (ic - in_channel_start) * params.kernel_h * params.kernel_w +
                                         kh * params.kernel_w +
                                         kw;

                        sum += input_data[input_idx] * weight_data[weight_idx];
                    }
                }
            }
        }

        // Add bias if present
        if (params.has_bias == 1) {
            sum += bias_data[oc];
        }

        // Write output
        uint output_idx = batch * params.out_channels * params.out_height * params.out_width +
                         oc * params.out_height * params.out_width +
                         out_y * params.out_width +
                         out_x;
        output_data[output_idx] = sum;
    }
}
