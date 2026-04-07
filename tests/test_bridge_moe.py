"""Bridge exposes fused MoE alongside other grilly_core ops (lazy device + shaders)."""

from grilly.backend import _bridge


def test_bridge_moe_functions_exported():
    for name in (
        "moe_upload",
        "moe_forward",
        "moe_backward",
        "moe_update_weights",
        "moe_release",
    ):
        assert hasattr(_bridge, name), f"missing _bridge.{name}"
