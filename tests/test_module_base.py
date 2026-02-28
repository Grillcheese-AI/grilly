"""
Comprehensive CPU-only tests for grilly.nn.module.Module base class.

Tests the Module API: parameters, train/eval, state_dict, device management,
gpu_mode, zero_grad, repr, and __call__ forwarding.
"""

import numpy as np
import pytest

try:
    from grilly.nn.module import Module
    from grilly.nn.parameter import Parameter
except ImportError:
    pytest.skip("grilly not available", allow_module_level=True)


# ---------------------------------------------------------------------------
# Helper Module subclasses
# ---------------------------------------------------------------------------


class SimpleModule(Module):
    """Minimal module with a single parameter for testing."""

    def __init__(self):
        super().__init__()
        self.w = Parameter(np.ones((3, 3), dtype=np.float32))
        self.register_parameter("w", self.w)

    def forward(self, x):
        return x  # trivial forward, no GPU


class NestedModule(Module):
    """Module that contains a child SimpleModule."""

    def __init__(self):
        super().__init__()
        self.child = SimpleModule()
        self._modules["child"] = self.child

    def forward(self, x):
        return self.child(x)


class TwoParamModule(Module):
    """Module with two parameters for multi-param tests."""

    def __init__(self):
        super().__init__()
        self.weight = Parameter(np.random.randn(4, 4).astype(np.float32))
        self.bias = Parameter(np.zeros(4, dtype=np.float32))
        self.register_parameter("weight", self.weight)
        self.register_parameter("bias", self.bias)

    def forward(self, x):
        return x


class DeepNestedModule(Module):
    """Module with two levels of nesting for deep hierarchy tests."""

    def __init__(self):
        super().__init__()
        self.layer = NestedModule()
        self._modules["layer"] = self.layer
        self.extra = Parameter(np.ones((2,), dtype=np.float32))
        self.register_parameter("extra", self.extra)

    def forward(self, x):
        return self.layer(x)


class EmptyModule(Module):
    """Module with no parameters or children."""

    def forward(self, x):
        return x


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestModuleInit:
    """Test Module.__init__ defaults."""

    def test_training_default_true(self):
        """Newly created module should be in training mode."""
        m = Module()
        assert m.training is True

    def test_parameters_dict_empty(self):
        """_parameters should start empty."""
        m = Module()
        assert m._parameters == {}

    def test_modules_dict_empty(self):
        """_modules should start empty."""
        m = Module()
        assert m._modules == {}

    def test_buffers_dict_empty(self):
        """_buffers should start empty."""
        m = Module()
        assert m._buffers == {}

    def test_device_default(self):
        """Default device should be 'vulkan'."""
        m = Module()
        assert m._device == "vulkan"

    def test_grad_enabled_default(self):
        """Gradients should be enabled by default."""
        m = Module()
        assert m._grad_enabled is True

    def test_gpu_tensor_default_false(self):
        """GPU-resident output should be disabled by default."""
        m = Module()
        assert m._return_gpu_tensor is False

    def test_device_local_default_false(self):
        """Device-local VRAM mode should be disabled by default."""
        m = Module()
        assert m._use_device_local is False


class TestModuleParameters:
    """Test parameter registration, iteration, and named_parameters."""

    def test_register_parameter_adds_to_dict(self):
        """register_parameter should store the parameter in _parameters."""
        m = Module()
        p = Parameter(np.zeros((2, 2), dtype=np.float32))
        m.register_parameter("test_param", p)
        assert "test_param" in m._parameters

    def test_register_parameter_none_removes(self):
        """Registering None should remove an existing parameter."""
        m = Module()
        p = Parameter(np.zeros((2, 2), dtype=np.float32))
        m.register_parameter("p", p)
        assert "p" in m._parameters
        m.register_parameter("p", None)
        assert "p" not in m._parameters

    def test_register_parameter_converts_ndarray(self):
        """Registering a plain ndarray should convert to Parameter."""
        m = Module()
        arr = np.ones((2,), dtype=np.float32)
        m.register_parameter("arr", arr)
        assert isinstance(m._parameters["arr"], Parameter)

    def test_parameters_yields_parameter_objects(self):
        """parameters() should yield Parameter instances."""
        m = SimpleModule()
        params = list(m.parameters())
        assert len(params) == 1
        assert isinstance(params[0], Parameter)

    def test_parameters_shape(self):
        """parameters() should yield the correct shapes."""
        m = SimpleModule()
        params = list(m.parameters())
        assert params[0].shape == (3, 3)

    def test_named_parameters_yields_tuples(self):
        """named_parameters() should yield (name, param) tuples."""
        m = SimpleModule()
        named = list(m.named_parameters())
        assert len(named) == 1
        name, param = named[0]
        assert name == "w"
        assert isinstance(param, Parameter)

    def test_multiple_parameters(self):
        """Module with two parameters should yield both."""
        m = TwoParamModule()
        params = list(m.parameters())
        assert len(params) == 2

    def test_multiple_named_parameters(self):
        """named_parameters() should yield correct names for multiple params."""
        m = TwoParamModule()
        names = {name for name, _ in m.named_parameters()}
        assert names == {"weight", "bias"}

    def test_nested_parameters_collected(self):
        """parameters() should collect parameters from child modules."""
        m = NestedModule()
        params = list(m.parameters())
        assert len(params) == 1
        assert params[0].shape == (3, 3)

    def test_nested_named_parameters_dot_notation(self):
        """named_parameters() should use dot notation for nested params."""
        m = NestedModule()
        named = list(m.named_parameters())
        assert len(named) == 1
        name, _ = named[0]
        assert name == "child.w"

    def test_deep_nested_named_parameters(self):
        """Deep nesting should produce multi-dot names."""
        m = DeepNestedModule()
        named = dict(m.named_parameters())
        assert "extra" in named
        assert "layer.child.w" in named
        assert len(named) == 2

    def test_deep_nested_parameters_count(self):
        """parameters() should collect from all levels."""
        m = DeepNestedModule()
        params = list(m.parameters())
        assert len(params) == 2

    def test_empty_module_no_parameters(self):
        """Module with no registered parameters yields nothing."""
        m = EmptyModule()
        assert list(m.parameters()) == []
        assert list(m.named_parameters()) == []


class TestModuleTrainEval:
    """Test train/evaluation mode switching."""

    def test_train_sets_training_true(self):
        """train() should set training=True."""
        m = SimpleModule()
        m.training = False
        m.train()
        assert m.training is True

    def test_eval_sets_training_false(self):
        """Module.eval() should set training=False."""
        m = SimpleModule()
        m.eval()
        assert m.training is False

    def test_train_false_sets_training_false(self):
        """train(False) should set training=False like Module.eval()."""
        m = SimpleModule()
        m.train(False)
        assert m.training is False

    def test_train_returns_self(self):
        """train() should return the module for chaining."""
        m = SimpleModule()
        result = m.train()
        assert result is m

    def test_eval_returns_self(self):
        """Module.eval() should return the module for chaining."""
        m = SimpleModule()
        result = m.eval()
        assert result is m

    def test_train_propagates_to_children(self):
        """train() should propagate mode to child modules."""
        m = NestedModule()
        m.eval()
        assert m.child.training is False
        m.train()
        assert m.child.training is True

    def test_eval_propagates_to_children(self):
        """Module.eval() should propagate mode to child modules."""
        m = NestedModule()
        m.eval()
        assert m.child.training is False

    def test_train_propagates_deeply(self):
        """train/eval should propagate through multiple nesting levels."""
        m = DeepNestedModule()
        m.eval()
        assert m.layer.child.training is False
        m.train()
        assert m.layer.child.training is True


class TestModuleZeroGrad:
    """Test zero_grad clears or initializes gradients."""

    def test_zero_grad_with_existing_grad(self):
        """zero_grad should zero out existing grad arrays."""
        m = SimpleModule()
        param = list(m.parameters())[0]
        param.grad = np.ones_like(param, dtype=np.float32)
        m.zero_grad()
        np.testing.assert_array_equal(param.grad, np.zeros_like(param))

    def test_zero_grad_initializes_none_grad(self):
        """zero_grad should create zero grad for params with grad=None."""
        m = SimpleModule()
        param = list(m.parameters())[0]
        assert param.grad is None
        m.zero_grad()
        assert param.grad is not None
        np.testing.assert_array_equal(param.grad, np.zeros((3, 3), dtype=np.float32))

    def test_zero_grad_correct_shape(self):
        """Initialized grad should match parameter shape."""
        m = TwoParamModule()
        m.zero_grad()
        for name, param in m.named_parameters():
            assert param.grad is not None
            assert param.grad.shape == param.shape

    def test_zero_grad_preserves_dtype(self):
        """Grad should be float32."""
        m = SimpleModule()
        m.zero_grad()
        param = list(m.parameters())[0]
        assert param.grad.dtype == np.float32

    def test_zero_grad_nested_modules(self):
        """zero_grad should zero gradients in child modules too."""
        m = NestedModule()
        child_param = list(m.child.parameters())[0]
        child_param.grad = np.ones((3, 3), dtype=np.float32) * 5.0
        m.zero_grad()
        np.testing.assert_array_equal(child_param.grad, np.zeros((3, 3), dtype=np.float32))

    def test_zero_grad_idempotent(self):
        """Calling zero_grad twice should be safe."""
        m = SimpleModule()
        m.zero_grad()
        m.zero_grad()
        param = list(m.parameters())[0]
        np.testing.assert_array_equal(param.grad, np.zeros((3, 3), dtype=np.float32))


class TestModuleStateDict:
    """Test state_dict and load_state_dict."""

    def test_state_dict_returns_dict(self):
        """state_dict should return a plain dict."""
        m = SimpleModule()
        sd = m.state_dict()
        assert isinstance(sd, dict)

    def test_state_dict_correct_keys(self):
        """state_dict keys should match registered parameter names."""
        m = SimpleModule()
        sd = m.state_dict()
        assert "w" in sd

    def test_state_dict_correct_values(self):
        """state_dict values should be numpy arrays matching param values."""
        m = SimpleModule()
        sd = m.state_dict()
        np.testing.assert_array_equal(sd["w"], np.ones((3, 3), dtype=np.float32))

    def test_state_dict_is_copy(self):
        """state_dict values should be copies, not references."""
        m = SimpleModule()
        sd = m.state_dict()
        sd["w"][0, 0] = 999.0
        param = list(m.parameters())[0]
        assert param[0, 0] != 999.0

    def test_state_dict_multiple_params(self):
        """state_dict should include all parameters."""
        m = TwoParamModule()
        sd = m.state_dict()
        assert "weight" in sd
        assert "bias" in sd

    def test_state_dict_nested_module(self):
        """Nested module should appear as a sub-dict in state_dict."""
        m = NestedModule()
        sd = m.state_dict()
        assert "child" in sd
        assert isinstance(sd["child"], dict)
        assert "w" in sd["child"]

    def test_load_state_dict_updates_values(self):
        """load_state_dict should update parameter values."""
        m = SimpleModule()
        new_sd = {"w": np.full((3, 3), 42.0, dtype=np.float32)}
        m.load_state_dict(new_sd)
        np.testing.assert_array_equal(m._parameters["w"], np.full((3, 3), 42.0))

    def test_load_state_dict_roundtrip(self):
        """Saving and loading state_dict should preserve values."""
        m1 = TwoParamModule()
        sd = m1.state_dict()
        m2 = TwoParamModule()
        m2.load_state_dict(sd)
        for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
            assert n1 == n2
            np.testing.assert_array_almost_equal(np.array(p1), np.array(p2))

    def test_load_state_dict_nested(self):
        """load_state_dict should update nested module parameters."""
        m = NestedModule()
        new_val = np.full((3, 3), 7.0, dtype=np.float32)
        sd = {"child": {"w": new_val}}
        m.load_state_dict(sd)
        child_param = list(m.child.parameters())[0]
        np.testing.assert_array_almost_equal(np.array(child_param), new_val)

    def test_load_state_dict_ignores_extra_keys(self):
        """Extra keys in state_dict should be ignored without error."""
        m = SimpleModule()
        sd = {"w": np.ones((3, 3), dtype=np.float32), "nonexistent": np.zeros(5)}
        m.load_state_dict(sd)  # should not raise

    def test_state_dict_empty_module(self):
        """Empty module should return empty state_dict."""
        m = EmptyModule()
        sd = m.state_dict()
        assert sd == {}


class TestModuleDevice:
    """Test device management: to(), cpu(), cuda(), vulkan()."""

    def test_to_cpu(self):
        """to('cpu') should set device and return self."""
        m = Module()
        result = m.to("cpu")
        assert m._device == "cpu"
        assert result is m

    def test_to_vulkan(self):
        """to('vulkan') should set device."""
        m = Module()
        m.to("cpu")
        m.to("vulkan")
        assert m._device == "vulkan"

    def test_to_cuda(self):
        """to('cuda') should set device."""
        m = Module()
        m.to("cuda")
        assert m._device == "cuda"

    def test_to_llama_cpp(self):
        """to('llama-cpp') should set device."""
        m = Module()
        m.to("llama-cpp")
        assert m._device == "llama-cpp"

    def test_to_none_returns_self(self):
        """to(None) should be a no-op and return self."""
        m = Module()
        original_device = m._device
        result = m.to(None)
        assert result is m
        assert m._device == original_device

    def test_to_unknown_device_ignored(self):
        """Unsupported device name should not change the current device."""
        m = Module()
        original_device = m._device
        m.to("tpu")
        assert m._device == original_device

    def test_cpu_method(self):
        """cpu() convenience method should set device to 'cpu'."""
        m = Module()
        result = m.cpu()
        assert m._device == "cpu"
        assert result is m

    def test_cuda_method(self):
        """cuda() convenience method should set device to 'cuda'."""
        m = Module()
        result = m.cuda()
        assert m._device == "cuda"
        assert result is m

    def test_vulkan_method(self):
        """vulkan() convenience method should set device to 'vulkan'."""
        m = Module()
        m.cpu()
        result = m.vulkan()
        assert m._device == "vulkan"
        assert result is m

    def test_llama_cpp_method(self):
        """llama_cpp() convenience method should set device to 'llama-cpp'."""
        m = Module()
        result = m.llama_cpp()
        assert m._device == "llama-cpp"
        assert result is m

    def test_to_case_insensitive(self):
        """to() should accept uppercase device names."""
        m = Module()
        m.to("CPU")
        assert m._device == "cpu"


class TestModuleGpuMode:
    """Test gpu_mode flag propagation."""

    def test_gpu_mode_enable(self):
        """gpu_mode(True) should set _return_gpu_tensor."""
        m = SimpleModule()
        m.gpu_mode(True)
        assert m._return_gpu_tensor is True

    def test_gpu_mode_disable(self):
        """gpu_mode(False) should unset _return_gpu_tensor."""
        m = SimpleModule()
        m.gpu_mode(True)
        m.gpu_mode(False)
        assert m._return_gpu_tensor is False

    def test_gpu_mode_sets_device_local(self):
        """gpu_mode(True) should also set _use_device_local by default."""
        m = SimpleModule()
        m.gpu_mode(True)
        assert m._use_device_local is True

    def test_gpu_mode_disable_clears_device_local(self):
        """gpu_mode(False) should clear _use_device_local."""
        m = SimpleModule()
        m.gpu_mode(True)
        m.gpu_mode(False)
        assert m._use_device_local is False

    def test_gpu_mode_without_device_local(self):
        """gpu_mode(True, device_local=False) should not set _use_device_local."""
        m = SimpleModule()
        m.gpu_mode(True, device_local=False)
        assert m._return_gpu_tensor is True
        assert m._use_device_local is False

    def test_gpu_mode_propagates_to_children(self):
        """gpu_mode should propagate to child modules."""
        m = NestedModule()
        m.gpu_mode(True)
        assert m.child._return_gpu_tensor is True
        assert m.child._use_device_local is True

    def test_gpu_mode_false_propagates_to_children(self):
        """gpu_mode(False) should propagate to child modules."""
        m = NestedModule()
        m.gpu_mode(True)
        m.gpu_mode(False)
        assert m.child._return_gpu_tensor is False
        assert m.child._use_device_local is False

    def test_gpu_mode_propagates_deeply(self):
        """gpu_mode should propagate through all nesting levels."""
        m = DeepNestedModule()
        m.gpu_mode(True)
        assert m.layer.child._return_gpu_tensor is True
        assert m.layer.child._use_device_local is True

    def test_gpu_mode_returns_self(self):
        """gpu_mode() should return self for chaining."""
        m = SimpleModule()
        result = m.gpu_mode(True)
        assert result is m


class TestModuleRepr:
    """Test __repr__ output."""

    def test_repr_includes_class_name(self):
        """repr should include the class name."""
        m = Module()
        assert "Module" in repr(m)

    def test_repr_simple_module(self):
        """repr of a subclass should use the subclass name."""
        m = SimpleModule()
        assert "SimpleModule" in repr(m)

    def test_repr_nested_module(self):
        """repr of NestedModule should use its class name."""
        m = NestedModule()
        assert "NestedModule" in repr(m)

    def test_repr_is_string(self):
        """repr should return a string."""
        m = Module()
        assert isinstance(repr(m), str)


class TestModuleCall:
    """Test __call__ invokes forward()."""

    def test_call_invokes_forward(self):
        """Calling the module should invoke forward()."""
        m = SimpleModule()
        x = np.ones((2, 3), dtype=np.float32)
        result = m(x)
        np.testing.assert_array_equal(result, x)

    def test_call_converts_input_to_float32(self):
        """__call__ should convert float64 input to float32."""
        m = SimpleModule()
        x = np.ones((2, 3), dtype=np.float64)
        result = m(x)
        assert result.dtype == np.float32

    def test_call_preserves_float32_input(self):
        """__call__ should preserve already-float32 input."""
        m = SimpleModule()
        x = np.ones((2, 3), dtype=np.float32)
        result = m(x)
        assert result.dtype == np.float32

    def test_call_nested_module(self):
        """Calling a nested module should pass through child forward."""
        m = NestedModule()
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = m(x)
        np.testing.assert_array_equal(result, x)

    def test_forward_not_implemented_base(self):
        """Calling base Module.forward() should raise NotImplementedError."""
        m = Module()
        with pytest.raises(NotImplementedError):
            m(np.ones(3, dtype=np.float32))

    def test_call_empty_module(self):
        """EmptyModule should return input unchanged."""
        m = EmptyModule()
        x = np.array([1.0, 2.0], dtype=np.float32)
        result = m(x)
        np.testing.assert_array_equal(result, x)
