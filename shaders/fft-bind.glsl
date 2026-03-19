#version 450

// ── Grilly Block-Code FFT Binder ─────────────────────────────────────────
// Computes Circular Convolution (Binding) or Correlation (Unbinding)
// entirely in GPU shared memory using Radix-2 FFT butterflies.
//
// Workgroup size: exactly l (e.g., 128 threads for l=128).
// Each Workgroup processes one of the k blocks.
//
// Push constants:
//   k         - number of blocks
//   l         - block length (must be power of 2, e.g., 128)
//   is_unbind - 0 = Bind (Convolution), 1 = Unbind (Correlation)

layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

// Buffers
layout(set = 0, binding = 0, std430) readonly buffer BufA { float dataA[]; };
layout(set = 0, binding = 1, std430) readonly buffer BufB { float dataB[]; };
layout(set = 0, binding = 2, std430) writeonly buffer BufOut { float dataOut[]; };

// Push Constants
layout(push_constant) uniform PushParams {
    uint k;
    uint l;
    uint is_unbind;
} params;

// Shared memory for the workgroup (fits in 32KB local memory)
// vec2 acts as complex numbers: x = real, y = imaginary
const uint MAX_L = 256;
shared vec2 shared_A[MAX_L];
shared vec2 shared_B[MAX_L];

const float PI = 3.14159265358979323846;

// ── Helper: Complex Multiplication ──
vec2 complex_mul(vec2 a, vec2 b) {
    return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// ── Helper: Bit Reversal for FFT Sorting ──
uint reverseBits(uint x, uint bits) {
    uint r = 0;
    for (uint i = 0; i < bits; i++) {
        if ((x & (1u << i)) != 0) {
            r |= (1u << ((bits - 1) - i));
        }
    }
    return r;
}

// ── Core Radix-2 FFT Butterfly pass ──
void compute_fft(inout vec2 arr[MAX_L], uint tid, uint l, uint log2l) {
    for (uint stage = 1; stage <= log2l; stage++) {
        uint step = 1u << stage;
        uint half_step = step >> 1;

        uint pair_idx = tid % half_step;
        uint block_start = (tid / half_step) * step;
        uint top_idx = block_start + pair_idx;
        uint bot_idx = top_idx + half_step;

        float angle = -2.0 * PI * float(pair_idx) / float(step);
        vec2 twiddle = vec2(cos(angle), sin(angle));

        barrier();

        vec2 top = arr[top_idx];
        vec2 bot = arr[bot_idx];
        vec2 t = complex_mul(twiddle, bot);

        barrier();

        arr[top_idx] = top + t;
        arr[bot_idx] = top - t;

        barrier();
    }
}

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint bid = gl_WorkGroupID.x;

    if (tid >= params.l || bid >= params.k) return;

    uint log2l = uint(log2(float(params.l)));
    uint global_idx = bid * params.l + tid;

    // 1. Bit-Reversed Load into Shared Memory
    uint rev_tid = reverseBits(tid, log2l);
    shared_A[rev_tid] = vec2(dataA[bid * params.l + tid], 0.0);
    shared_B[rev_tid] = vec2(dataB[bid * params.l + tid], 0.0);
    barrier();

    // 2. Forward FFT on A and B
    compute_fft(shared_A, tid, params.l, log2l);
    compute_fft(shared_B, tid, params.l, log2l);

    // 3. Point-wise Complex Multiplication (Frequency Domain)
    barrier();
    vec2 valA = shared_A[tid];
    vec2 valB = shared_B[tid];

    if (params.is_unbind == 1) {
        // Unbinding (Correlation): A * conj(B)
        valB.y = -valB.y;
    }

    vec2 valC = complex_mul(valA, valB);

    // Setup for IFFT via conjugation method: IFFT(X) = conj(FFT(conj(X))) / N
    uint rev_tid_ifft = reverseBits(tid, log2l);
    shared_A[rev_tid_ifft] = vec2(valC.x, -valC.y);
    barrier();

    // 4. Inverse FFT (reuse forward FFT function)
    compute_fft(shared_A, tid, params.l, log2l);

    // 5. Final Conjugate, Scale, and Write to Global Memory
    barrier();
    vec2 final_val = shared_A[tid];

    dataOut[global_idx] = final_val.x / float(params.l);
}
