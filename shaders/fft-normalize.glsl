#version 450

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Real part (FFT output, will be input to IFFT)
layout(set = 0, binding = 0) buffer Real {
    float data_real[];
};

// Imaginary part (FFT output, will be input to IFFT)
layout(set = 0, binding = 1) buffer Imag {
    float data_imag[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint total_elements;
    uint N;              // Signal length (for normalization)
};

void main() {
    uint gID = gl_GlobalInvocationID.x;
    
    if (gID >= total_elements) {
        return;
    }
    
    // IFFT normalization: divide by N
    // This is done after butterfly operations complete
    data_real[gID] /= float(N);
    data_imag[gID] /= float(N);
}
