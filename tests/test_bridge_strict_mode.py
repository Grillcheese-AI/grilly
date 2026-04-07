import numpy as np


def test_bridge_strict_mode_raises(monkeypatch):
    from grilly.backend import _bridge

    class DummyCore:
        @staticmethod
        def linear(dev, x, weight, bias):
            raise RuntimeError("forced failure")

    monkeypatch.setattr(_bridge, "_core", DummyCore())
    monkeypatch.setattr(_bridge, "_get_device", lambda: object())
    monkeypatch.setattr(_bridge, "_BRIDGE_STRICT", True)
    _bridge.reset_fallback_stats()

    x = np.ones((2, 4), dtype=np.float32)
    w = np.ones((3, 4), dtype=np.float32)
    try:
        _bridge.linear(x, w, None)
        assert False, "Expected strict bridge failure to raise"
    except RuntimeError as e:
        assert "GRILLY_BRIDGE_STRICT=1" in str(e)


def test_bridge_fallback_stats_increment(monkeypatch):
    from grilly.backend import _bridge

    class DummyCore:
        @staticmethod
        def relu(dev, x):
            raise RuntimeError("forced failure")

    monkeypatch.setattr(_bridge, "_core", DummyCore())
    monkeypatch.setattr(_bridge, "_get_device", lambda: object())
    monkeypatch.setattr(_bridge, "_BRIDGE_STRICT", False)
    _bridge.reset_fallback_stats()

    x = np.array([-1.0, 2.0], dtype=np.float32)
    out = _bridge.relu(x)
    assert out is None
    stats = _bridge.get_fallback_stats()
    assert stats.get("relu", 0) == 1
