#version 450

// ── Batch VSA Logic Operator ────────────────────────────────────────
//
// Applies N different logic operators to one state vector in parallel.
// Used for the "Hypothesize" phase of latent reasoning.
//
// Each workgroup applies one operator to the working memory via XOR
// (binding in bipolar VSA). The output is N hypothesis vectors, each
// representing a different "thought" — a possible next state.
//
// Dispatch: workgroups_x = num_ops
//           Each workgroup's 256 threads handle a stripe of the vector.
//
// In CubeMind, logic operators are mapped to Rubik's Cube group
// transformations: U, D, L, R, F, B moves and their compositions.
// Binding with an operator "rotates" the working memory through the
// VSA geometry, exploring logical neighborhoods.

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer StateBuf {
    uint working_memory[];
};

layout(set = 0, binding = 1) readonly buffer OperatorBuf {
    uint op_pool[];
};

layout(set = 0, binding = 2) writeonly buffer HypothesisBuf {
    uint hypotheses[];
};

layout(push_constant) uniform PushConsts {
    uint words_per_vec;   // dim / 32 (e.g., 320)
    uint num_ops;         // Number of logic operators to apply
    uint _pad0;
    uint _pad1;
};

void main() {
    uint op_idx = gl_WorkGroupID.x;
    uint local_idx = gl_LocalInvocationID.x;

    if (op_idx >= num_ops) return;

    uint op_offset = op_idx * words_per_vec;
    uint hyp_offset = op_idx * words_per_vec;

    // XOR binding: hypothesis = working_memory ^ operator
    // In bipolar VSA, this is equivalent to element-wise multiplication.
    // The result is a new state in the permutation group orbit.
    for (uint i = local_idx; i < words_per_vec; i += 256) {
        hypotheses[hyp_offset + i] = working_memory[i] ^ op_pool[op_offset + i];
    }
}
