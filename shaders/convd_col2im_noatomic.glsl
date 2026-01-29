#version 450

/*
 * Conv2d col2im - Non-atomic version
 *
 * Input:  cols (K_dim, N_cols), flattened
 * Output: grad_input (N, C_in, H_in, W_in) - NCHW
 *
 * K_dim = C_in * kernel_h * kernel_w
 * N_cols = N * H_out * W_out
 *
 * Strategy: Each thread handles ONE output pixel and accumulates from all
 * cols entries that map to it. No race conditions, no atomics.
 */

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(binding = 0) readonly buffer ColsBuffer {
    float cols[];
};

layout(binding = 1) writeonly buffer GradInputBuffer {
    float grad_input[];
};

layout(push_constant) uniform PushConstants {
    uint batch_size;
    uint in_channels;
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
    uint dilation_h;
    uint dilation_w;
} params;

void main() {
    // Each thread handles one output pixel: (n, ic, in_y, in_x)
    uint global_idx = gl_GlobalInvocationID.x +
                      gl_GlobalInvocationID.y * gl_NumWorkGroups.x * gl_WorkGroupSize.x;

    uint total_output_pixels = params.batch_size * params.in_channels *
                                params.in_height * params.in_width;

    if (global_idx >= total_output_pixels) {
        return;
    }

    // Decode global_idx into (n, ic, in_y, in_x)
    uint n = global_idx / (params.in_channels * params.in_height * params.in_width);
    uint rem1 = global_idx % (params.in_channels * params.in_height * params.in_width);
    uint ic = rem1 / (params.in_height * params.in_width);
    uint rem2 = rem1 % (params.in_height * params.in_width);
    uint in_y = rem2 / params.in_width;
    uint in_x = rem2 % params.in_width;

    float sum = 0.0;
    uint N_cols = params.batch_size * params.out_height * params.out_width;

    // Loop over all kernel positions (kh, kw) that could contribute to this input pixel
    for (uint kh = 0; kh < params.kernel_h; kh++) {
        for (uint kw = 0; kw < params.kernel_w; kw++) {
            // Given kernel position (kh, kw), which output positions (out_y, out_x) use this input pixel?
            // Equation: in_y = out_y * stride_h + kh * dilation_h - padding_h
            // Solve for out_y: out_y = (in_y + padding_h - kh * dilation_h) / stride_h

            int numerator_y = int(in_y) + int(params.padding_h) - int(kh * params.dilation_h);
            int numerator_x = int(in_x) + int(params.padding_w) - int(kw * params.dilation_w);

            // Check if divisible by stride
            if (numerator_y % int(params.stride_h) != 0 || numerator_x % int(params.stride_w) != 0) {
                continue;
            }

            int out_y = numerator_y / int(params.stride_h);
            int out_x = numerator_x / int(params.stride_w);

            // Check bounds
            if (out_y < 0 || out_y >= int(params.out_height) ||
                out_x < 0 || out_x >= int(params.out_width)) {
                continue;
            }

            // Compute indices into cols array
            // k_idx = ic * kernel_h * kernel_w + kh * kernel_w + kw
            uint k_idx = ic * params.kernel_h * params.kernel_w + kh * params.kernel_w + kw;

            // col_idx = n * out_height * out_width + out_y * out_width + out_x
            uint col_idx = n * params.out_height * params.out_width +
                           uint(out_y) * params.out_width + uint(out_x);

            // cols is stored as (K_dim, N_cols) row-major
            uint cols_idx = k_idx * N_cols + col_idx;

            sum += cols[cols_idx];
        }
    }

    // Write result
    grad_input[global_idx] = sum;
}
