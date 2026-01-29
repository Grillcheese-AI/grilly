#version 450

/*
 * AvgPool2d Backward Pass
 *
 * Dispatches over INPUT positions (avoids race conditions, no atomics needed).
 * Each input position accumulates gradients from all output positions that used it.
 */

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(binding = 0) readonly buffer GradOutputBuffer {
    float grad_output[];
};

layout(binding = 1) writeonly buffer GradInputBuffer {
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
    // Calculate INPUT position (dispatch over input space)
    uint batch = gl_GlobalInvocationID.z / params.channels;
    uint channel = gl_GlobalInvocationID.z % params.channels;
    uint in_y = gl_GlobalInvocationID.y;
    uint in_x = gl_GlobalInvocationID.x;

    if (batch >= params.batch_size || in_y >= params.in_height || in_x >= params.in_width) {
        return;
    }

    float grad_sum = 0.0;

    // Find all output positions that used this input position
    // For each possible output position, check if it would read from this input
    for (uint out_y = 0; out_y < params.out_height; out_y++) {
        for (uint out_x = 0; out_x < params.out_width; out_x++) {
            // Determine if this output position reads from our input position
            // Output (out_y, out_x) reads from input window:
            //   [out_y*stride_h - padding_h : out_y*stride_h - padding_h + kernel_h)
            //   [out_x*stride_w - padding_w : out_x*stride_w - padding_w + kernel_w)

            int window_start_y = int(out_y * params.stride_h) - int(params.padding_h);
            int window_end_y = window_start_y + int(params.kernel_h);
            int window_start_x = int(out_x * params.stride_w) - int(params.padding_w);
            int window_end_x = window_start_x + int(params.kernel_w);

            // Check if our input position (in_y, in_x) is in this window
            if (int(in_y) >= window_start_y && int(in_y) < window_end_y &&
                int(in_x) >= window_start_x && int(in_x) < window_end_x) {

                // This output used our input - accumulate its gradient
                uint grad_out_idx = batch * params.channels * params.out_height * params.out_width +
                                   channel * params.out_height * params.out_width +
                                   out_y * params.out_width +
                                   out_x;

                float grad = grad_output[grad_out_idx];

                // Count valid positions in this output's window
                uint count = 0;
                for (uint kh = 0; kh < params.kernel_h; kh++) {
                    for (uint kw = 0; kw < params.kernel_w; kw++) {
                        int iy = int(out_y * params.stride_h + kh) - int(params.padding_h);
                        int ix = int(out_x * params.stride_w + kw) - int(params.padding_w);

                        if (iy >= 0 && iy < int(params.in_height) &&
                            ix >= 0 && ix < int(params.in_width)) {
                            count++;
                        } else if (params.count_include_pad == 1) {
                            count++;
                        }
                    }
                }

                float grad_per_element = (count > 0) ? (grad / float(count)) : 0.0;
                grad_sum += grad_per_element;
            }
        }
    }

    // Write final gradient
    uint grad_in_idx = batch * params.channels * params.in_height * params.in_width +
                      channel * params.in_height * params.in_width +
                      in_y * params.in_width +
                      in_x;

    grad_input[grad_in_idx] = grad_sum;
}
