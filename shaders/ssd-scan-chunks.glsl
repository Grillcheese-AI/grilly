#version 450

// SSD Chunked Scan Shader
//
// Bidirectional within-chunk linear recurrence for SSD (State Space Duality)
// training mode.  Each invocation processes one (batch_chunk, feature) lane
// across chunk_size timesteps.
//
// Forward:   s[j] = decay * s[j-1] + u[j],   j = 0 .. chunk_size-1
// Backward:  g[j] = decay * g[j+1] + gs[j],  j = chunk_size-1 .. 0
//
// Inter-chunk state propagation is handled on the host (tiny O(T/C) scan).
//
// Bindings:
//   0  input_data   (batch_chunks * chunk_size * features)  read-only
//   1  decay_data   (features)                              read-only
//   2  init_state   (batch_chunks * features)               read-only
//   3  output_data  (batch_chunks * chunk_size * features)  write-only
//   4  carry_out    (batch_chunks * features)               write-only

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly  buffer InputBuf   { float input_data[]; };
layout(set = 0, binding = 1) readonly  buffer DecayBuf   { float decay_data[]; };
layout(set = 0, binding = 2) readonly  buffer InitBuf    { float init_state[]; };
layout(set = 0, binding = 3) writeonly buffer OutputBuf  { float output_data[]; };
layout(set = 0, binding = 4) writeonly buffer CarryBuf   { float carry_out[]; };

layout(push_constant) uniform PushConstants {
    uint batch_chunks;   // B * num_chunks
    uint chunk_size;     // C (64 typical)
    uint features;       // D
    uint is_backward;    // 0 = forward, 1 = backward (reverse direction)
} pc;

void main() {
    uint lane = gl_GlobalInvocationID.x;
    uint total_lanes = pc.batch_chunks * pc.features;

    if (lane >= total_lanes) {
        return;
    }

    uint bc = lane / pc.features;   // batch_chunk index
    uint f  = lane % pc.features;   // feature index

    float d = decay_data[f];
    float state = init_state[bc * pc.features + f];

    // Base offset for this (batch_chunk, feature) lane in the flat array
    // Layout: [batch_chunk, timestep, feature]  (row-major)
    uint base   = bc * pc.chunk_size * pc.features + f;
    uint stride = pc.features;

    if (pc.is_backward == 0u) {
        // ---- Forward scan ----
        for (uint j = 0; j < pc.chunk_size; ++j) {
            uint idx = base + j * stride;
            float u = input_data[idx];
            state = d * state + u;
            output_data[idx] = state;
        }
    } else {
        // ---- Backward scan (reverse direction) ----
        for (int j = int(pc.chunk_size) - 1; j >= 0; --j) {
            uint idx = base + uint(j) * stride;
            float gs = input_data[idx];
            state = d * state + gs;
            output_data[idx] = state;
        }
    }

    // Write carry-out: the final state after processing this chunk
    carry_out[bc * pc.features + f] = state;
}
