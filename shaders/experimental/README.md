# Experimental VSA Shaders

GPU-accelerated shaders for Vector Symbolic Architecture operations.

## Compiling Shaders

These GLSL shaders must be compiled to SPIR-V format before use:

```bash
# Create SPV directory if it doesn't exist
mkdir -p shaders/experimental/spv

# Compile each shader (compute shaders require -fshader-stage=compute)
glslc -fshader-stage=compute shaders/experimental/vsa-bind.glsl -o shaders/experimental/spv/vsa-bind.spv
glslc -fshader-stage=compute shaders/experimental/vsa-bundle.glsl -o shaders/experimental/spv/vsa-bundle.spv
glslc -fshader-stage=compute shaders/experimental/vsa-similarity-batch.glsl -o shaders/experimental/spv/vsa-similarity-batch.spv
glslc -fshader-stage=compute shaders/experimental/vsa-fft-convolve.glsl -o shaders/experimental/spv/vsa-fft-convolve.spv
```

Or compile all at once:

```bash
for shader in vsa-bind vsa-bundle vsa-similarity-batch vsa-fft-convolve; do
    glslc -fshader-stage=compute shaders/experimental/${shader}.glsl -o shaders/experimental/spv/${shader}.spv
done
```

## Shader Descriptions

- **vsa-bind.glsl**: Element-wise multiplication for bipolar binding (O(d))
- **vsa-bundle.glsl**: Superposition with majority voting (O(d))
- **vsa-similarity-batch.glsl**: Parallel cosine similarity computation (O(V*d))
- **vsa-fft-convolve.glsl**: FFT-based circular convolution for HRR (O(d log d))

## Requirements

- `glslc` (GLSL compiler from Vulkan SDK)
- Vulkan SDK installed
