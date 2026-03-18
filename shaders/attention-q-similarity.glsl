#version 450
layout(local_size_x = 256) in;

// Input: queries of shape (batch, seq_len, head_dim)
layout(set=0, binding=0) readonly buffer Queries { float queries[]; };
// Output: q_similarity per batch element, shape (batch,)
layout(set=0, binding=1) writeonly buffer Output { float q_sim[]; };

layout(push_constant) uniform Params {
    uint batch_size;
    uint seq_len;
    uint head_dim;
};

void main() {
    uint batch_idx = gl_GlobalInvocationID.x;
    if (batch_idx >= batch_size) return;

    float total_sim = 0.0;
    uint count = seq_len - 1;
    if (count == 0) {
        q_sim[batch_idx] = 1.0;
        return;
    }

    uint base = batch_idx * seq_len * head_dim;

    for (uint t = 0; t < count; t++) {
        // Compute cosine similarity between q_t and q_{t+1}
        float dot_prod = 0.0;
        float norm_a = 0.0;
        float norm_b = 0.0;

        uint offset_a = base + t * head_dim;
        uint offset_b = base + (t + 1) * head_dim;

        for (uint d = 0; d < head_dim; d++) {
            float a = queries[offset_a + d];
            float b = queries[offset_b + d];
            dot_prod += a * b;
            norm_a += a * a;
            norm_b += b * b;
        }

        float denom = sqrt(norm_a) * sqrt(norm_b);
        total_sim += (denom > 1e-8) ? (dot_prod / denom) : 0.0;
    }

    q_sim[batch_idx] = total_sim / float(count);
}
