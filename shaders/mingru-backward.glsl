#version 450

/*
 * MinGRU Fused Activation + Causal Scan (backward)
 *
 * Inputs:
 *   DH (grad_h): (B, S, D)
 *   G, V, D: (B, S, D) - Original gate projections
 *   H: (B, S, D) - Results of forward pass
 *
 * Outputs:
 *   DG, DV, DD: (B, S, D) - Gradients for projections
 */

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0) readonly  buffer GradH  { float DH[]; };
layout(binding = 1) readonly  buffer GateG  { float G[];  };
layout(binding = 2) readonly  buffer GateV  { float V[];  };
layout(binding = 3) readonly  buffer GateD  { float D[];  };
layout(binding = 4) readonly  buffer State  { float H[];  };
layout(binding = 5) writeonly buffer GradG  { float DG[]; };
layout(binding = 6) writeonly buffer GradV  { float DV[]; };
layout(binding = 7) writeonly buffer GradD  { float DD[]; };

layout(push_constant) uniform PushConstants {
    uint seq_len;
    uint hidden_dim;
} params;

float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

float sigmoid_grad(float s) {
    return s * (1.0 - s);
}

void main() {
    uint d = gl_GlobalInvocationID.x;  // hidden dim index
    uint b = gl_WorkGroupID.y;         // batch index

    if (d >= params.hidden_dim) return;

    uint stride_t = params.hidden_dim;
    uint base     = b * params.seq_len * params.hidden_dim + d;

    float dh_next = 0.0;
    
    // Reverse time loop
    for (int t = int(params.seq_len) - 1; t >= 0; --t) {
        uint idx = base + uint(t) * stride_t;
        
        float g_t = G[idx];
        float v_t = V[idx];
        float d_t = D[idx];
        
        float sig_g = sigmoid(g_t);
        float tanh_v = tanh(v_t);
        float sig_d = sigmoid(d_t);
        
        float x_scan_t = sig_g * tanh_v;
        float a_t      = 0.05 + 0.9 * sig_d;
        
        // Gradient from output + gradient from next time step
        float dh_total = DH[idx] + dh_next;
        
        // 1. Grad w.r.t x_scan
        float d_x_scan = dh_total;
        DG[idx] = d_x_scan * sigmoid_grad(sig_g) * tanh_v;
        DV[idx] = d_x_scan * sig_g * (1.0 - tanh_v * tanh_v);
        
        // 2. Grad w.r.t a_t
        float h_prev = (t > 0) ? H[idx - stride_t] : 0.0;
        float d_a_t = dh_total * h_prev;
        DD[idx] = d_a_t * 0.9 * sigmoid_grad(sig_d);
        
        // 3. Propagate to previous step
        dh_next = dh_total * a_t;
    }
}
