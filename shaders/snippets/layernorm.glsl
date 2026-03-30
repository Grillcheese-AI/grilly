// Snippet: LayerNorm (per-row normalization)
// Operates on shared memory tile — caller must set up LDS.
// w = scale, b = bias, dim = row length
void op_layernorm(inout float val, float w, float b,
                   float mean, float inv_std) {
    val = (val - mean) * inv_std * w + b;
}
