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

// Output magnitude
layout(set = 0, binding = 2) buffer Magnitude {
    float mag[];
};

// Output phase (optional)
layout(set = 0, binding = 3) buffer Phase {
    float phase[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint total_elements;
    uint compute_phase;   // 1 = compute phase, 0 = magnitude only
};

void main() {
    uint gID = gl_GlobalInvocationID.x;
    
    if (gID >= total_elements) {
        return;
    }
    
    float re = data_real[gID];
    float im = data_imag[gID];
    
    // Magnitude: sqrt(re^2 + im^2)
    mag[gID] = sqrt(re * re + im * im);
    
    // Phase: atan2(im, re)
    if (compute_phase == 1) {
        phase[gID] = atan(im, re);
    }
}
