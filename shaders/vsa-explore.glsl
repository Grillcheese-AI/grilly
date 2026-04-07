#version 450
#extension GL_EXT_shader_atomic_float : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable

// ═══════════════════════════════════════════════════════════════════════
// VSA Exploration Shader — GPU-side rule discovery sandbox
//
// Uses atomic int workspace as a "thinking space" for exploring
// VSA relationships. The workspace is initialized to 0, used for
// trial bind/unbind/bundle operations, then results are read out
// and workspace reset to 0 — zero allocation overhead.
//
// Dispatch with: (n_trials / 256) workgroups
//
// Each thread explores one candidate rule by:
//   1. Binding/unbinding observation vectors
//   2. Computing similarity to codebook entries
//   3. Atomically accumulating votes into the workspace
//   4. The workspace accumulates a "transition histogram"
//      that reveals the dominant rule structure
// ═══════════════════════════════════════════════════════════════════════

layout(local_size_x = 256) in;

// Push constants
layout(push_constant) uniform PushConstants {
    uint n_trials;        // Number of exploration trials
    uint n_states;        // Codebook size
    uint k;               // Blocks per vector
    uint l;               // Block length
    uint seq_len;         // Observation sequence length
    uint mode;            // 0=bind, 1=unbind, 2=bundle, 3=circular_shift
    float temperature;    // Softmax temperature for similarity
};

// Observation sequence: seq_len vectors, each k*l floats
layout(std430, binding = 0) readonly buffer Observations {
    float obs[];  // [seq_len * k * l]
};

// Codebook: n_states vectors, each k*l floats
layout(std430, binding = 1) readonly buffer Codebook {
    float codebook[];  // [n_states * k * l]
};

// Workspace: atomic int accumulator for transition counts
// Shape: [n_states * n_states] — transition_count[from][to]
layout(std430, binding = 2) buffer Workspace {
    int workspace[];  // [n_states * n_states], init to 0
};

// Output: best transition matrix (float, normalized).
// NOTE: not ``writeonly`` — the shader also reads this buffer as a scratch
// space during exploration (see similarity() below, and the block_bind
// "reusing output buffer temporarily" path).
// ``output`` is a reserved word in recent glslang — renamed to output_data.
layout(std430, binding = 3) buffer Output {
    float output_data[];  // [n_states * n_states]
};

// ── Per-block circular convolution (bind) ────────────────────
// Operates on a single block of length l
void block_bind(uint block_offset_a, uint block_offset_b,
                uint block_offset_out, uint block_len) {
    // Direct circular convolution: out[i] = sum_m a[m] * b[(i-m) mod l]
    for (uint i = 0; i < block_len; i++) {
        float sum = 0.0;
        for (uint m = 0; m < block_len; m++) {
            uint idx_b = (i + block_len - m) % block_len;
            sum += obs[block_offset_a + m] * obs[block_offset_b + idx_b];
        }
        // Store in shared memory (reusing output buffer temporarily)
        output_data[block_offset_out + i] = sum;
    }
}

// ── Similarity: dot product between vector and codebook entry ─
float similarity(uint vec_offset, uint cb_entry, uint dim) {
    float dot = 0.0;
    for (uint i = 0; i < dim; i++) {
        dot += output_data[vec_offset + i] * codebook[cb_entry * dim + i];
    }
    return dot / float(k);  // Normalize by number of blocks
}

// ── Main exploration kernel ──────────────────────────────────
void main() {
    uint tid = gl_GlobalInvocationID.x;
    if (tid >= n_trials) return;

    uint dim = k * l;

    // Each thread explores one consecutive pair in the sequence
    uint t = tid % (seq_len - 1);
    uint obs_a_offset = t * dim;
    uint obs_b_offset = (t + 1) * dim;

    // Find which codebook entry obs[t] is closest to
    float best_sim_a = -1e30;
    uint best_state_a = 0;
    for (uint s = 0; s < n_states; s++) {
        float sim = 0.0;
        for (uint i = 0; i < dim; i++) {
            sim += obs[obs_a_offset + i] * codebook[s * dim + i];
        }
        sim /= float(k);
        if (sim > best_sim_a) {
            best_sim_a = sim;
            best_state_a = s;
        }
    }

    // Find which codebook entry obs[t+1] is closest to
    float best_sim_b = -1e30;
    uint best_state_b = 0;
    for (uint s = 0; s < n_states; s++) {
        float sim = 0.0;
        for (uint i = 0; i < dim; i++) {
            sim += obs[obs_b_offset + i] * codebook[s * dim + i];
        }
        sim /= float(k);
        if (sim > best_sim_b) {
            best_sim_b = sim;
            best_state_b = s;
        }
    }

    // Atomically increment transition count: A[from][to]++
    atomicAdd(workspace[best_state_a * n_states + best_state_b], 1);
}
