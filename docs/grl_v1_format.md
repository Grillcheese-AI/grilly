# GRL v1 checkpoint format (`.grl`)

Grilly’s native, pickle-free checkpoint container for model weights, optimizer metadata, and training scalars.

## Layout

1. **Header** (64 bytes): magic `GRLY`, `uint16` format version (1), `uint16` flags, `uint32` reserved, then `uint64` offsets/lengths for metadata JSON, tensor index JSON, and raw payload.
2. **Metadata JSON** (UTF-8): `schema: "grilly.checkpoint.v1"`, `framework: "grilly"`, optional keys such as `training_step`, `best_ppl`, `step`, `epoch`, and `extra`.
3. **Tensor index JSON**: ordered array of `{name, dtype, shape, offset, length}` with `offset`/`length` relative to the **start of the payload** section.
4. **Payload**: concatenated C-contiguous row-major tensor bytes (little-endian scalars in index).

## dtypes

Index encodes dtypes as `f32`, `f16`, `i64`, `i32`, `u8`.

## API

- Write: `grilly.utils.grl_checkpoint.save_grl(path, state_dict, metadata=...)`.
- Read: `grilly.utils.grl_checkpoint.load_grl(path, map_location=...)`.
- Torch-style: `grilly.torch_api.save` / `grilly.torch_api.load` (`.grl` only).

## Versioning

Format version **1** is fixed in `FORMAT_VERSION` in `utils/grl_checkpoint.py` and the C++ reader/writer. Future versions must bump the header version and preserve a migration path.
