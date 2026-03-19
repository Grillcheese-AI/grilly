# VSA Superposition Crosstalk — I-RAVEN Grid Bottleneck

**Date:** 2026-03-19
**Context:** CubeMind I-RAVEN benchmark at 86.1% overall

## The Pattern

Single-entity configs (Center, L-R, U-D, O-IC) hit 96-100%.
Multi-entity grid configs (2x2, 3x3, O-IG) plateau at 67-77%.

## Root Cause: Capacity Saturation

When bundling 9 objects into one panel vector:
Panel = (Obj1 x Pos1) + (Obj2 x Pos2) + ... + (Obj9 x Pos9)

Each bundle adds noise. With 9 objects in d=2048, the SNR drops below
the threshold needed for reliable unbinding. The similarity scores blur
together, and the softmax can't confidently pick the correct candidate.

## Two Fixes

### 1. Expand the Space (brute force)
Increase from k=16, l=128 (d=2048) to k=32, l=256 (d=8192).
Exponentially more capacity. Vulkan shaders barely notice the size increase.

### 2. Hierarchical Bundling (elegant)
Instead of flat bundling all 9 objects:
- Bundle objects into rows first: Row1 = (Obj11 x X1) + (Obj21 x X2) + (Obj31 x X3)
- Then bundle rows into panel: Panel = (Row1 x Y1) + (Row2 x Y2) + (Row3 x Y3)
- Creates cleaner geometric structure for unbinding

## Current Workaround
Mode aggregation of entity attributes loses per-entity information.
The integer detectors work on aggregated values but get wrong predictions
34% of the time when mode is ambiguous.

## Relationship to Differentiable Pipeline
The CNN -> blockSoftmax -> bind/unbind pipeline designed earlier can learn
hierarchical bundling automatically via backprop. The VSA algebra is its
own adjoint, so the gradient naturally discovers the right decomposition.
