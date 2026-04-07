"""Bridge exposes fused VSA-LM alongside other grilly_core ops (lazy device + shaders)."""

from grilly.backend import _bridge


def test_bridge_vsa_lm_functions_exported():
    for name in (
        "vsa_lm_upload",
        "vsa_lm_forward",
        "vsa_lm_backward",
        "vsa_lm_update_weights",
        "vsa_lm_release",
    ):
        assert hasattr(_bridge, name), f"missing _bridge.{name}"
