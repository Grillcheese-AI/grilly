#version 450

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Position indices (batch, seq_len)
layout(set = 0, binding = 0) readonly buffer Positions {
    float positions[];
};

// Output positional encodings (batch, seq_len, embedding_dim)
layout(set = 0, binding = 1) buffer Output {
    float output_data[];
};

// Learnable theta phase offsets (embedding_dim)
layout(set = 0, binding = 2) readonly buffer ThetaOffsets {
    float theta_phase_offsets[];
};

// Learnable gamma phase offsets (embedding_dim)
layout(set = 0, binding = 3) readonly buffer GammaOffsets {
    float gamma_phase_offsets[];
};

// Amplitude modulation parameters (embedding_dim)
layout(set = 0, binding = 4) readonly buffer AmplitudeModulation {
    float amp_mod[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint batch_size;
    uint seq_len;
    uint embedding_dim;
    float theta_freq;      // Theta frequency (e.g., 8 Hz)
    float gamma_freq;      // Gamma frequency (e.g., 40 Hz)
    uint max_seq_len;      // For stable normalization
};

const float PI = 3.14159265359;

void main() {
    uint gID = gl_GlobalInvocationID.x;
    uint total_elements = batch_size * seq_len * embedding_dim;
    
    if (gID >= total_elements) {
        return;
    }
    
    // Decode indices: [batch, seq, embed]
    uint batch_idx = gID / (seq_len * embedding_dim);
    uint remainder = gID % (seq_len * embedding_dim);
    uint seq_idx = remainder / embedding_dim;
    uint embed_idx = remainder % embedding_dim;
    
    // Get position
    uint pos_idx = batch_idx * seq_len + seq_idx;
    float pos = positions[pos_idx];
    
    // Normalize position to [0, 2π] using max_seq_len for stability
    float denom = float(max(max_seq_len - 1, 1));
    float normalized_pos = (pos / denom) * (2.0 * PI);
    
    // Get learnable offsets and amplitude
    float theta_offset = theta_phase_offsets[embed_idx];
    float gamma_offset = gamma_phase_offsets[embed_idx];
    float amplitude = amp_mod[embed_idx];
    
    // === Theta Phase Encoding ===
    float theta_phase = normalized_pos + theta_offset;
    float theta_encoding = sin(theta_phase);
    
    // === Gamma Phase Encoding ===
    // Frequency ratio for phase coupling
    float freq_ratio = gamma_freq / theta_freq;
    float gamma_phase = (normalized_pos * freq_ratio) + gamma_offset;
    
    // === Phase-Amplitude Coupling (PAC) ===
    // Gamma amplitude modulated by theta phase (biological PAC)
    // Gamma is strongest at theta peaks (cosine modulation)
    float gamma_amplitude = (cos(theta_phase) + 1.0) * 0.5;
    float gamma_encoding = gamma_amplitude * sin(gamma_phase);
    
    // === Combine Encodings ===
    // Theta carries slow positional info, gamma carries fast detailed info
    float combined = (theta_encoding + 0.5 * gamma_encoding) * amplitude;
    
    output_data[gID] = combined;
}
