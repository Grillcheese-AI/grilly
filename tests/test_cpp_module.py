"""
Tests for C++ Module base class with pybind11 trampoline.
Validates Python subclassing, parameter registration, state_dict, and train/eval modes.
"""

import numpy as np
import pytest

try:
    from grilly_core import Module, Parameter, Tensor

    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False

pytestmark = pytest.mark.skipif(not CPP_AVAILABLE, reason="C++ backend not available")


class TestModuleSubclassing:
    """Test Python subclassing of C++ Module via trampoline."""

    def test_identity_forward(self):
        """Python Module subclass should be callable with forward()."""

        class Identity(Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x

        mod = Identity()
        x = Tensor.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        y = mod(x)
        np.testing.assert_array_equal(y.numpy(), [1.0, 2.0, 3.0])

    def test_forward_with_computation(self):
        """Python forward can modify the tensor."""

        class ScaleModule(Module):
            def __init__(self, scale):
                super().__init__()
                self.scale = scale

            def forward(self, x):
                arr = x.numpy() * self.scale
                return Tensor.from_numpy(arr)

        mod = ScaleModule(2.0)
        x = Tensor.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        y = mod(x)
        np.testing.assert_array_equal(y.numpy(), [2.0, 4.0, 6.0])

    def test_call_dispatches_to_forward(self):
        """__call__ should dispatch to forward()."""

        class Negate(Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return Tensor.from_numpy(-x.numpy())

        mod = Negate()
        x = Tensor.from_numpy(np.array([1.0, -2.0], dtype=np.float32))
        y = mod(x)
        np.testing.assert_array_equal(y.numpy(), [-1.0, 2.0])


class TestModuleParameters:
    """Test parameter registration and retrieval."""

    def test_register_parameter(self):
        """register_parameter should make param available via parameters()."""

        class Linear(Module):
            def __init__(self):
                super().__init__()
                w = Parameter(Tensor.from_numpy(np.ones((3, 4), dtype=np.float32)))
                self.register_parameter("weight", w)

            def forward(self, x):
                return x

        mod = Linear()
        params = mod.parameters()
        assert len(params) == 1
        assert params[0].shape == [3, 4]

    def test_named_parameters(self):
        """named_parameters should return (name, param) pairs."""

        class TwoParam(Module):
            def __init__(self):
                super().__init__()
                self.register_parameter(
                    "weight", Parameter(Tensor.zeros([3, 4]))
                )
                self.register_parameter(
                    "bias", Parameter(Tensor.zeros([4]))
                )

            def forward(self, x):
                return x

        mod = TwoParam()
        named = mod.named_parameters()
        names = [n for n, p in named]
        assert "weight" in names
        assert "bias" in names

    def test_register_module(self):
        """register_module should include child module parameters."""

        class Child(Module):
            def __init__(self):
                super().__init__()
                self.register_parameter(
                    "w", Parameter(Tensor.zeros([2, 3]))
                )

            def forward(self, x):
                return x

        class Parent(Module):
            def __init__(self):
                super().__init__()
                self.register_module("child", Child())

            def forward(self, x):
                return x

        mod = Parent()
        params = mod.parameters()
        assert len(params) == 1

        named = mod.named_parameters()
        assert named[0][0] == "child.w"

    def test_nested_modules_parameters(self):
        """Parameters from deeply nested modules should be collected."""

        class Leaf(Module):
            def __init__(self, name):
                super().__init__()
                self.register_parameter(
                    name, Parameter(Tensor.zeros([2]))
                )

            def forward(self, x):
                return x

        class Mid(Module):
            def __init__(self):
                super().__init__()
                self.register_module("leaf", Leaf("w"))

            def forward(self, x):
                return x

        class Root(Module):
            def __init__(self):
                super().__init__()
                self.register_module("mid", Mid())

            def forward(self, x):
                return x

        mod = Root()
        named = mod.named_parameters()
        assert len(named) == 1
        assert named[0][0] == "mid.leaf.w"


class TestModuleTrainEval:
    """Test train/eval mode switching."""

    def test_default_is_training(self):
        """Module should default to training mode."""

        class M(Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x

        mod = M()
        assert mod.training is True

    def test_eval_sets_not_training(self):
        """eval() should set training=False."""

        class M(Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x

        mod = M()
        mod.eval()
        assert mod.training is False

    def test_train_restores_training(self):
        """train() should restore training mode."""

        class M(Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x

        mod = M()
        mod.eval()
        mod.train()
        assert mod.training is True

    def test_train_propagates_to_children(self):
        """train/eval should propagate to child modules."""
        from grilly_core import IFNode

        parent_mod = IFNode(step_mode="s")
        parent_mod.eval()
        assert parent_mod.training is False
        parent_mod.train()
        assert parent_mod.training is True


class TestModuleStateDict:
    """Test state_dict serialization."""

    def test_state_dict_contains_parameters(self):
        """state_dict should contain all registered parameters as numpy arrays."""

        class Net(Module):
            def __init__(self):
                super().__init__()
                self.register_parameter(
                    "weight",
                    Parameter(Tensor.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32))),
                )

            def forward(self, x):
                return x

        mod = Net()
        sd = mod.state_dict()
        assert "weight" in sd
        np.testing.assert_array_equal(sd["weight"], [1.0, 2.0, 3.0])

    def test_load_state_dict(self):
        """load_state_dict should restore parameter values."""

        class Net(Module):
            def __init__(self):
                super().__init__()
                self.register_parameter(
                    "weight",
                    Parameter(Tensor.from_numpy(np.zeros(3, dtype=np.float32))),
                )

            def forward(self, x):
                return x

        mod = Net()
        new_state = {"weight": np.array([4.0, 5.0, 6.0], dtype=np.float32)}
        mod.load_state_dict(new_state)
        sd = mod.state_dict()
        np.testing.assert_array_equal(sd["weight"], [4.0, 5.0, 6.0])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
