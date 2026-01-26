#version 450

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Real part
layout(set = 0, binding = 0) readonly buffer Real {
    float data_real[];
};

// Imaginary part
layout(set = 0, binding = 1) readonly buffer Imag {
    float data_imag[];
};

// Power spectrum output
layout(set = 0, binding = 2) buffer PowerSpectrum {
    float power[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint total_elements;
    uint scale_by_n;     // 1 = divide by N (for normalization), 0 = raw power
    uint N;              // Signal length
};

void main() {
    uint gID = gl_GlobalInvocationID.x;
    
    if (gID >= total_elements) {
        return;
    }
    
    float re = data_real[gID];
    float im = data_imag[gID];
    
    // Power: |X|^2 = re^2 + im^2
    float pwr = re * re + im * im;
    
    // Optional normalization
    if (scale_by_n == 1) {
        pwr /= float(N);
    }
    
    power[gID] = pwr;
}
