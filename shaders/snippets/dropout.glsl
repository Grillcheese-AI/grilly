// Snippet: Dropout with PCG hash for deterministic masking
uint pcg_hash_drop(uint state) {
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float op_dropout(float x, uint idx, uint seed, float p) {
    uint h = pcg_hash_drop(idx ^ pcg_hash_drop(seed));
    float r = float(h) / 4294967296.0;
    return r < p ? 0.0 : x / (1.0 - p);
}
