#version 450

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Input real values (batch, N)
layout(set = 0, binding = 0) readonly buffer InputReal {
    float input_real[];
};

// Output real part (batch, N)
layout(set = 0, binding = 1) buffer OutputReal {
    float output_real[];
};

// Output imaginary part (batch, N)
layout(set = 0, binding = 2) buffer OutputImag {
    float output_imag[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint batch_size;
    uint N;              // Signal length (must be power of 2)
};

// Bit reversal function
uint bitReverse(uint x, uint log2N) {
    uint result = 0;
    for (uint i = 0; i < log2N; i++) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}

void main() {
    uint gID = gl_GlobalInvocationID.x;
    uint total_elements = batch_size * N;
    
    if (gID >= total_elements) {
        return;
    }
    
    uint batch_idx = gID / N;
    uint n = gID % N;
    
    // Compute log2(N)
    uint log2N = 0;
    uint temp = N;
    while (temp > 1) {
        temp >>= 1;
        log2N++;
    }
    
    // Bit-reversal permutation
    uint reversed_n = bitReverse(n, log2N);
    uint input_idx = batch_idx * N + reversed_n;
    
    // Copy with bit-reversal (initialize imaginary to 0)
    output_real[gID] = input_real[input_idx];
    output_imag[gID] = 0.0;
}
