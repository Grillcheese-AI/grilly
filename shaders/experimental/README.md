# Experimental VSA Shaders

GPU-accelerated shaders for Vector Symbolic Architecture operations.

## Compiling Shaders

These GLSL shaders must be compiled to SPIR-V format before use:

```bash
# Create SPV directory if it doesn't exist
mkdir -p shaders/experimental/spv

# Compile each shader (compute shaders require -fshader-stage=compute)
glslc -fshader-stage=compute shaders/experimental/vsa-bind.glsl -o shaders/experimental/spv/vsa-bind.spv
glslc -fshader-stage=compute shaders/experimental/vsa-bind-batch.glsl -o shaders/experimental/spv/vsa-bind-batch.spv
glslc -fshader-stage=compute shaders/experimental/vsa-bundle.glsl -o shaders/experimental/spv/vsa-bundle.spv
glslc -fshader-stage=compute shaders/experimental/vsa-bundle-batch.glsl -o shaders/experimental/spv/vsa-bundle-batch.spv
glslc -fshader-stage=compute shaders/experimental/vsa-similarity-batch.glsl -o shaders/experimental/spv/vsa-similarity-batch.spv
glslc -fshader-stage=compute shaders/experimental/vsa-resonator-step.glsl -o shaders/experimental/spv/vsa-resonator-step.spv
glslc -fshader-stage=compute shaders/experimental/vsa-fft-convolve.glsl -o shaders/experimental/spv/vsa-fft-convolve.spv
```

Or compile all at once:

```bash
for shader in vsa-bind vsa-bind-batch vsa-bundle vsa-bundle-batch vsa-similarity-batch vsa-resonator-step vsa-fft-convolve; do
    glslc -fshader-stage=compute shaders/experimental/${shader}.glsl -o shaders/experimental/spv/${shader}.spv
done
```

## Shader Descriptions

- **vsa-bind.glsl**: Element-wise multiplication for bipolar binding (O(d))
- **vsa-bind-batch.glsl**: Batched bipolar binding (O(B*d))
- **vsa-bundle.glsl**: Superposition with majority voting (O(d))
- **vsa-bundle-batch.glsl**: Batched superposition with majority voting (O(B*d))
- **vsa-similarity-batch.glsl**: Parallel cosine similarity computation (O(V*d))
- **vsa-resonator-step.glsl**: Codebook projection for resonator step (O(V*d))
- **vsa-fft-convolve.glsl**: FFT-based circular convolution for HRR (O(d log d))

## Requirements

- `glslc` (GLSL compiler from Vulkan SDK)
- Vulkan SDK installed
