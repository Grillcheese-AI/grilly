#version 450

/*
 * 2D Convolution Backward Pass - Gradient w.r.t. Weights
 *
 * Computes gradient of loss w.r.t. weights.
 * This accumulates gradients across all batch samples.
 *
 * grad_output shape: (batch, out_channels, out_h, out_w)
 * input shape: (batch, in_channels, in_h, in_w)
 * grad_weight shape: (out_channels, in_channels/groups, kernel_h, kernel_w)
 * grad_bias shape: (out_channels)
 */

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(binding = 0) readonly buffer GradOutputBuffer {
    float grad_output[];
};

layout(binding = 1) readonly buffer InputBuffer {
    float input_data[];
};

layout(binding = 2) buffer GradWeightBuffer {
    float grad_weight[];  // Accumulates gradients
};

layout(binding = 3) buffer GradBiasBuffer {
    float grad_bias[];  // Optional - accumulates bias gradients
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

// Shared memory for reduction (accumulate gradients within workgroup)
shared float shared_grad_weight[256];  // 16x16 workgroup
shared float shared_grad_bias[256];

void main() {
    uint local_idx = gl_LocalInvocationIndex;

    // Calculate weight position
    uint oc = gl_GlobalInvocationID.y;  // Output channel
    uint weight_spatial = gl_GlobalInvocationID.x;  // Flattened (ic, kh, kw)

    uint in_channels_per_group = params.in_channels / params.groups;
    uint total_weight_spatial = in_channels_per_group * params.kernel_h * params.kernel_w;

    // Initialize shared memory
    shared_grad_weight[local_idx] = 0.0;
    if (params.has_bias == 1) {
        shared_grad_bias[local_idx] = 0.0;
    }

    // Bounds check
    if (oc >= params.out_channels || weight_spatial >= total_weight_spatial) {
        barrier();
        return;
    }

    // Decode weight position
    uint ic_in_group = weight_spatial / (params.kernel_h * params.kernel_w);
    uint kernel_spatial = weight_spatial % (params.kernel_h * params.kernel_w);
    uint kh = kernel_spatial / params.kernel_w;
    uint kw = kernel_spatial % params.kernel_w;

    // Determine group and absolute input channel
    uint group = oc * params.groups / params.out_channels;
    uint ic = group * in_channels_per_group + ic_in_group;

    float grad_w_sum = 0.0;
    float grad_b_sum = 0.0;

    // Accumulate over batch and spatial dimensions
    for (uint batch = 0; batch < params.batch_size; batch++) {
        for (uint out_y = 0; out_y < params.out_height; out_y++) {
            for (uint out_x = 0; out_x < params.out_width; out_x++) {
                // Calculate corresponding input position
                int in_y = int(out_y * params.stride_h + kh * params.dilation_h) - int(params.padding_h);
                int in_x = int(out_x * params.stride_w + kw * params.dilation_w) - int(params.padding_w);

                // Check if in valid input range
                if (in_y >= 0 && in_y < int(params.in_height) &&
                    in_x >= 0 && in_x < int(params.in_width)) {

                    // Calculate indices
                    uint grad_out_idx = batch * params.out_channels * params.out_height * params.out_width +
                                       oc * params.out_height * params.out_width +
                                       out_y * params.out_width +
                                       out_x;

                    uint input_idx = batch * params.in_channels * params.in_height * params.in_width +
                                    ic * params.in_height * params.in_width +
                                    uint(in_y) * params.in_width +
                                    uint(in_x);

                    grad_w_sum += grad_output[grad_out_idx] * input_data[input_idx];
                }
            }
        }

        // Accumulate bias gradient (sum over all positions for this output channel)
        if (params.has_bias == 1 && ic_in_group == 0 && kh == 0 && kw == 0) {
            for (uint out_y = 0; out_y < params.out_height; out_y++) {
                for (uint out_x = 0; out_x < params.out_width; out_x++) {
                    uint grad_out_idx = batch * params.out_channels * params.out_height * params.out_width +
                                       oc * params.out_height * params.out_width +
                                       out_y * params.out_width +
                                       out_x;
                    grad_b_sum += grad_output[grad_out_idx];
                }
            }
        }
    }

    // Write gradient to global memory
    uint weight_idx = oc * in_channels_per_group * params.kernel_h * params.kernel_w +
                     ic_in_group * params.kernel_h * params.kernel_w +
                     kh * params.kernel_w +
                     kw;
    grad_weight[weight_idx] = grad_w_sum;

    if (params.has_bias == 1 && ic_in_group == 0 && kh == 0 && kw == 0) {
        grad_bias[oc] = grad_b_sum;
    }
}
