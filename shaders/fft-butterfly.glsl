#version 450

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Real part (in-place)
layout(set = 0, binding = 0) buffer Real {
    float data_real[];
};

// Imaginary part (in-place)
layout(set = 0, binding = 1) buffer Imag {
    float data_imag[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint batch_size;
    uint N;              // Signal length (power of 2)
    uint stage;          // Current FFT stage (0 to log2(N)-1)
    uint inverse;        // 0 = forward FFT, 1 = inverse FFT
};

const float PI = 3.14159265359;

void main() {
    uint gID = gl_GlobalInvocationID.x;
    uint total_pairs = batch_size * N / 2;
    
    if (gID >= total_pairs) {
        return;
    }
    
    uint batch_idx = gID / (N / 2);
    uint pair_idx = gID % (N / 2);
    
    // Butterfly parameters for current stage
    uint m = 1 << (stage + 1);  // 2^(stage+1)
    uint m_half = m >> 1;        // m/2
    
    // Which butterfly group and position within group
    uint group = pair_idx / m_half;
    uint pos = pair_idx % m_half;
    
    // Butterfly element indices
    uint k = group * m + pos;
    uint idx1 = batch_idx * N + k;
    uint idx2 = idx1 + m_half;
    
    // Twiddle factor: W = exp(-2πi * pos / m) for forward, exp(+2πi * pos / m) for inverse
    float angle = 2.0 * PI * float(pos) / float(m);
    if (inverse == 1) {
        angle = -angle;
    }
    
    float cos_w = cos(angle);
    float sin_w = sin(angle);
    
    // Read values
    float a_real = data_real[idx1];
    float a_imag = data_imag[idx1];
    float b_real = data_real[idx2];
    float b_imag = data_imag[idx2];
    
    // Complex multiplication: t = W * b
    float t_real = cos_w * b_real - sin_w * b_imag;
    float t_imag = cos_w * b_imag + sin_w * b_real;
    
    // Butterfly operation
    // out1 = a + t
    // out2 = a - t
    data_real[idx1] = a_real + t_real;
    data_imag[idx1] = a_imag + t_imag;
    data_real[idx2] = a_real - t_real;
    data_imag[idx2] = a_imag - t_imag;
}
