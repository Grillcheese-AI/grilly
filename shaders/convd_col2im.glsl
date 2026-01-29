#version 450
#extension GL_EXT_shader_atomic_float : enable

/*
 * Conv2d col2im (inverse of im2col) for backward-input:
 *
 * Input:  cols (K_dim, N_cols), flattened
 * Output: image_data (N, C_in, H_in, W_in) in NCHW
 *
 * K_dim  = C_in * kernel_h * kernel_w
 * N_cols = N * H_out * W_out
 *
 * Each thread takes one (k_idx, col_idx) element from cols and
 * atomically accumulates it into the corresponding input pixel.
 */

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(binding = 0) readonly buffer ColsBuffer {
    float cols[];
};

layout(binding = 1) buffer ImageBuffer {
    float image_data[];  // must be zero-initialized before dispatch
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
    uint k_idx   = gl_GlobalInvocationID.x; // 0 .. K_dim-1
    uint col_idx = gl_GlobalInvocationID.y; // 0 .. N_cols-1

    uint K_dim   = params.in_channels * params.kernel_h * params.kernel_w;
    uint N_cols  = params.batch_size * params.out_height * params.out_width;

    if (k_idx >= K_dim || col_idx >= N_cols) {
        return;
    }

    // Decode k_idx into (ic, kh, kw)
    uint ic = k_idx / (params.kernel_h * params.kernel_w);
    uint rem = k_idx % (params.kernel_h * params.kernel_w);
    uint kh  = rem / params.kernel_w;
    uint kw  = rem % params.kernel_w;

    // Decode col_idx into (n, out_y, out_x)
    uint n      = col_idx / (params.out_height * params.out_width);
    uint rem2   = col_idx % (params.out_height * params.out_width);
    uint out_y  = rem2 / params.out_width;
    uint out_x  = rem2 % params.out_width;

    // Compute input coordinates
    int in_y = int(out_y * params.stride_h + kh * params.dilation_h) - int(params.padding_h);
    int in_x = int(out_x * params.stride_w + kw * params.dilation_w) - int(params.padding_w);

    // Read value from cols
    uint cols_idx = k_idx * N_cols + col_idx;
    float val = cols[cols_idx];

    // Accumulate into image_data if within bounds
    if (in_y >= 0 && in_y < int(params.in_height) &&
        in_x >= 0 && in_x < int(params.in_width)) {

        uint image_idx =
            n * params.in_channels * params.in_height * params.in_width +
            ic * params.in_height * params.in_width +
            uint(in_y) * params.in_width +
            uint(in_x);

        // Atomic add for accumulation
        atomicAdd(image_data[image_idx], val);
    }
}
