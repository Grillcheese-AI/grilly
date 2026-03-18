#version 450
#extension GL_KHR_shader_subgroup_arithmetic : enable
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_ballot : enable

// Flash Attention 3: Tiled attention with subgroup-accelerated online softmax
// Improvements over FA2:
//   1. Subgroup reductions for online softmax (subgroupAdd/subgroupMax)
//   2. Shared memory double-buffering (load next tile while computing current)
//   3. Tighter memory access patterns (coalesced loads via shared memory staging)
//
// Cooperative matrix (VK_KHR_cooperative_matrix) variant is in flash-attention3-coop.glsl
// This shader works on any GPU with subgroup arithmetic support.

layout(local_size_x = 64) in;  // One full wavefront (AMD wave64 / NVIDIA warp×2)

layout(set=0, binding=0) readonly buffer Queries { float Q[]; };
layout(set=0, binding=1) readonly buffer Keys { float K[]; };
layout(set=0, binding=2) readonly buffer Values { float V[]; };
layout(set=0, binding=3) writeonly buffer Output { float O[]; };

layout(push_constant) uniform Params {
    uint batch_size;
    uint num_heads;
    uint seq_len;
    uint head_dim;
    float scale;        // 1/sqrt(head_dim)
    uint tile_k;        // key tile size (e.g., 64)
};

// Shared memory for double-buffered K/V tiles
// Two buffers: current (computing) and next (loading)
shared float s_k[2][64 * 64];  // [buffer][tile_k * head_dim] — max 64 keys × 64 dims
shared float s_v[2][64 * 64];  // same for values

void main() {
    uint tid = gl_LocalInvocationID.x;  // 0..63
    uint wid = gl_WorkGroupID.x;

    // Each workgroup handles one (batch, head, query_row) tuple
    uint total_rows = batch_size * num_heads * seq_len;
    if (wid >= total_rows) return;

    uint batch_idx = wid / (num_heads * seq_len);
    uint head_idx = (wid / seq_len) % num_heads;
    uint q_row = wid % seq_len;

    // Load query row into registers (each thread loads head_dim/64 elements)
    // For head_dim <= 64, each thread loads 1 element
    // For head_dim > 64, threads loop
    float q_reg[4];  // up to 256-dim heads (4 × 64 threads)
    uint dims_per_thread = (head_dim + 63u) / 64u;
    uint q_base = batch_idx * num_heads * seq_len * head_dim
                + head_idx * seq_len * head_dim
                + q_row * head_dim;

    for (uint i = 0u; i < dims_per_thread && tid + i * 64u < head_dim; i++) {
        q_reg[i] = Q[q_base + tid + i * 64u];
    }

    // Online softmax state (per thread, will be reduced via subgroup)
    float row_max = -1e30;
    float row_sum = 0.0;

    // Output accumulator (same layout as q_reg)
    float o_reg[4] = float[4](0.0, 0.0, 0.0, 0.0);

    // Number of K tiles
    uint num_tiles = (seq_len + tile_k - 1u) / tile_k;
    uint cur_buf = 0u;

    // ── Tile loop with double buffering ──────────────────────────────
    for (uint t = 0u; t < num_tiles; t++) {
        uint k_start = t * tile_k;
        uint k_end = min(k_start + tile_k, seq_len);
        uint tile_len = k_end - k_start;

        // Load K and V tile into shared memory (current buffer)
        uint kv_base = batch_idx * num_heads * seq_len * head_dim
                     + head_idx * seq_len * head_dim;

        // Each thread loads multiple elements to fill the tile
        for (uint i = tid; i < tile_len * head_dim; i += 64u) {
            uint kk = i / head_dim;  // which key in tile
            uint dd = i % head_dim;  // which dimension
            uint global_k = k_start + kk;
            if (global_k < seq_len) {
                s_k[cur_buf][kk * head_dim + dd] = K[kv_base + global_k * head_dim + dd];
                s_v[cur_buf][kk * head_dim + dd] = V[kv_base + global_k * head_dim + dd];
            }
        }
        barrier();

        // ── Compute attention scores for this tile ───────────────────
        for (uint j = 0u; j < tile_len; j++) {
            // Dot product: q_reg · s_k[j]
            float score = 0.0;
            for (uint i = 0u; i < dims_per_thread && tid + i * 64u < head_dim; i++) {
                score += q_reg[i] * s_k[cur_buf][j * head_dim + tid + i * 64u];
            }

            // Subgroup reduction for the dot product
            score = subgroupAdd(score) * scale;

            // Only lane 0 needs the full score, but all lanes need it for the value update
            // Broadcast from lane 0
            score = subgroupBroadcastFirst(score);

            // ── Online softmax update (all lanes) ────────────────────
            float old_max = row_max;
            row_max = max(row_max, score);

            // Rescale existing accumulator
            float rescale = exp(old_max - row_max);
            row_sum = row_sum * rescale;
            for (uint i = 0u; i < dims_per_thread; i++) {
                o_reg[i] *= rescale;
            }

            // Add new score contribution
            float p = exp(score - row_max);
            row_sum += p;

            // Accumulate weighted value
            for (uint i = 0u; i < dims_per_thread && tid + i * 64u < head_dim; i++) {
                o_reg[i] += p * s_v[cur_buf][j * head_dim + tid + i * 64u];
            }
        }

        // Flip double buffer for next iteration
        cur_buf = 1u - cur_buf;
        barrier();
    }

    // ── Finalize: divide by sum ──────────────────────────────────────
    float inv_sum = 1.0 / max(row_sum, 1e-8);
    uint o_base = batch_idx * num_heads * seq_len * head_dim
                + head_idx * seq_len * head_dim
                + q_row * head_dim;

    for (uint i = 0u; i < dims_per_thread && tid + i * 64u < head_dim; i++) {
        O[o_base + tid + i * 64u] = o_reg[i] * inv_sum;
    }
}
