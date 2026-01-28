#version 450

/*
 * AvgPool2d Forward Pass
 *
 * Computes average pooling over spatial dimensions.
 * Input: (batch, channels, height, width)
 * Output: (batch, channels, out_h, out_w)
 */

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(binding = 0) readonly buffer InputBuffer {
    float input_data[];
};

layout(binding = 1) writeonly buffer OutputBuffer {
    float output_data[];
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
    uint count_include_pad;  // 0 or 1
} params;

void main() {
    // Calculate output position
    uint batch = gl_GlobalInvocationID.z / params.channels;
    uint channel = gl_GlobalInvocationID.z % params.channels;
    uint out_y = gl_GlobalInvocationID.y;
    uint out_x = gl_GlobalInvocationID.x;

    // Bounds check
    if (batch >= params.batch_size || out_y >= params.out_height || out_x >= params.out_width) {
        return;
    }

    float sum = 0.0;
    uint count = 0;

    // Pool over kernel
    for (uint kh = 0; kh < params.kernel_h; kh++) {
        for (uint kw = 0; kw < params.kernel_w; kw++) {
            // Calculate input position
            int in_y = int(out_y * params.stride_h + kh) - int(params.padding_h);
            int in_x = int(out_x * params.stride_w + kw) - int(params.padding_w);

            // Check if in valid input range
            if (in_y >= 0 && in_y < int(params.in_height) &&
                in_x >= 0 && in_x < int(params.in_width)) {

                uint input_idx = batch * params.channels * params.in_height * params.in_width +
                                channel * params.in_height * params.in_width +
                                uint(in_y) * params.in_width +
                                uint(in_x);

                sum += input_data[input_idx];
                count++;
            } else if (params.count_include_pad == 1) {
                count++;
            }
        }
    }

    // Compute average
    float avg = (count > 0) ? (sum / float(count)) : 0.0;

    // Write output
    uint output_idx = batch * params.channels * params.out_height * params.out_width +
                     channel * params.out_height * params.out_width +
                     out_y * params.out_width +
                     out_x;

    output_data[output_idx] = avg;
}
