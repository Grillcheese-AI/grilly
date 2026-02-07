#version 450
// Row-wise argmax for a score matrix (rows x cols), stored row-major.
// One workgroup per row. Each thread scans strided columns and reduces to max.
//
// bindings:
// 0: scores (float) length rows*stride
// 1: out_indices (uint) length rows*k (or rows, if k=1)
// 2: out_values  (float) length rows*k (or rows)
//
// push constants:
//   uint rows
//   uint cols
//   uint stride   (>= cols)
//   uint out_offset  (rank * rows)

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer Scores {
    float scores[];
};

layout(set = 0, binding = 1) buffer OutIdx {
    uint out_idx[];
};

layout(set = 0, binding = 2) buffer OutVal {
    float out_val[];
};

layout(push_constant) uniform PushConsts {
    uint rows;
    uint cols;
    uint stride;
    uint out_offset;
} pc;

shared float sVal[256];
shared uint  sIdx[256];

void main() {
    uint row = gl_WorkGroupID.x;
    uint tid = gl_LocalInvocationID.x;

    if (row >= pc.rows) { return; }
    if (pc.cols == 0u) {
        if (tid == 0u) {
            out_idx[pc.out_offset + row] = uint(-1);
            out_val[pc.out_offset + row] = -3.402823466e38;
        }
        return;
    }

    uint base = row * pc.stride;

    float best = -3.402823466e38; // -FLT_MAX
    uint best_i = 0u;

    for (uint col = tid; col < pc.cols; col += 256u) {
        float v = scores[base + col];
        if (v > best) { best = v; best_i = col; }
    }

    sVal[tid] = best;
    sIdx[tid] = best_i;
    barrier();

    // parallel reduction
    for (uint offset = 128u; offset > 0u; offset >>= 1u) {
        if (tid < offset) {
            float v2 = sVal[tid + offset];
            uint  i2 = sIdx[tid + offset];
            if (v2 > sVal[tid]) { sVal[tid] = v2; sIdx[tid] = i2; }
        }
        barrier();
    }

    if (tid == 0u) {
        out_idx[pc.out_offset + row] = sIdx[0];
        out_val[pc.out_offset + row] = sVal[0];
    }
}
