#version 450

/*
 * 2D Convolution Forward Pass (optimized mapping)
 *
 * Computes 2D convolution with:
 * - Stride, padding, dilation
 * - Grouped convolutions
 * - Bias addition
 *
 * Input shape:  (batch, in_channels, in_h, in_w)
 * Weight shape: (out_channels, in_channels/groups, kernel_h, kernel_w)
 * Output shape: (batch, out_channels, out_h, out_w)
 */

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(binding = 0) readonly buffer InputBuffer {
    float input_data[];
};

layout(binding = 1) readonly buffer WeightBuffer {
    float weight_data[];
};

layout(binding = 2) readonly buffer BiasBuffer {
    float bias_data[];
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
    uint has_bias;
} params;

void main() {
    // Spatial position
    uint out_x = gl_GlobalInvocationID.x;
    uint out_y = gl_GlobalInvocationID.y;

    if (out_x >= params.out_width || out_y >= params.out_height) {
        return;
    }

    // Map z to (batch, oc)
    // We dispatch group_count_z = batch_size * out_channels
    uint z = gl_GlobalInvocationID.z;
    uint batch = z / params.out_channels;
    uint oc    = z % params.out_channels;

    if (batch >= params.batch_size || oc >= params.out_channels) {
        return;
    }

    // Groups
    uint in_channels_per_group  = params.in_channels / params.groups;
    uint out_channels_per_group = params.out_channels / params.groups;

    uint group = oc / out_channels_per_group;
    uint in_channel_start = group * in_channels_per_group;
    uint in_channel_end   = in_channel_start + in_channels_per_group;

    float sum = 0.0;

    // Anchor output position in input coordinates (top-left of receptive field)
    int base_y = int(out_y * params.stride_h) - int(params.padding_h);
    int base_x = int(out_x * params.stride_w) - int(params.padding_w);

    // Convolution loop
    for (uint ic = in_channel_start; ic < in_channel_end; ic++) {
        // Base index for this (batch, ic)
        uint input_base =
            batch * params.in_channels * params.in_height * params.in_width +
            ic * params.in_height * params.in_width;

        uint weight_base =
            oc * in_channels_per_group * params.kernel_h * params.kernel_w +
            (ic - in_channel_start) * params.kernel_h * params.kernel_w;

        for (uint kh = 0; kh < params.kernel_h; kh++) {
            int in_y = base_y + int(kh * params.dilation_h);
            if (in_y < 0 || in_y >= int(params.in_height)) {
                continue;
            }

            uint input_row = input_base + uint(in_y) * params.in_width;

            for (uint kw = 0; kw < params.kernel_w; kw++) {
                int in_x = base_x + int(kw * params.dilation_w);
                if (in_x < 0 || in_x >= int(params.in_width)) {
                    continue;
                }

                uint input_idx  = input_row + uint(in_x);
                uint weight_idx = weight_base + kh * params.kernel_w + kw;

                sum += input_data[input_idx] * weight_data[weight_idx];
            }
        }
    }

    if (params.has_bias == 1u) {
        sum += bias_data[oc];
    }

    uint output_idx =
        batch * params.out_channels * params.out_height * params.out_width +
        oc    * params.out_height * params.out_width +
        out_y * params.out_width +
        out_x;

    output_data[output_idx] = sum;
}
