#version 450
#extension GL_KHR_shader_subgroup_ballot : enable
#extension GL_KHR_shader_subgroup_vote : enable

// MoQE Hard Routing via Subgroup Ballots.
//
// Each token picks exactly one expert (0=4-bit, 1=8-bit).
// subgroupBallot + subgroupAny avoids warp divergence:
// if no lane needs an expert, the entire fetch is skipped.
//
// Dispatch: X = embedding_dim, Y = num_tokens

layout (local_size_x = 32) in;

layout(std430, binding = 0) readonly buffer RouterChoice { uint choices[]; };
layout(std430, binding = 1) readonly buffer Expert0     { float e0_weights[]; };
layout(std430, binding = 2) readonly buffer Expert1     { float e1_weights[]; };
layout(std430, binding = 3) writeonly buffer Output     { float out_emb[]; };

layout(push_constant) uniform Params {
    uint embedding_dim;
};

void main() {
    uint token_idx = gl_GlobalInvocationID.y;
    uint dim_idx = gl_GlobalInvocationID.x;

    uint my_expert = choices[token_idx];
    bool want_e0 = (my_expert == 0);
    bool want_e1 = (my_expert == 1);

    uvec4 ballot_e0 = subgroupBallot(want_e0);
    uvec4 ballot_e1 = subgroupBallot(want_e1);

    float final_value = 0.0;

    // Skip entire expert fetch if no lane needs it
    if (subgroupAny(want_e0)) {
        if (want_e0) {
            final_value = e0_weights[token_idx * embedding_dim + dim_idx];
        }
    }

    if (subgroupAny(want_e1)) {
        if (want_e1) {
            final_value = e1_weights[token_idx * embedding_dim + dim_idx];
        }
    }

    out_emb[token_idx * embedding_dim + dim_idx] = final_value;
}
