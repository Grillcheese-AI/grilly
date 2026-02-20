"""Tests for Phase 9: ANN2SNN Conversion"""

import numpy as np
import pytest


class TestBNFusion:
    """Test Conv+BN fusion."""

    def test_fuse_conv_bn_numerically_equivalent(self):
        """Fused Conv+BN should produce same output as unfused."""
        from grilly.nn.conv import Conv2d
        from grilly.nn.module import Module
        from grilly.nn.normalization import BatchNorm2d
        from grilly.nn.snn_ann2snn import Converter

        # Create a simple model using _modules for forward
        class SimpleModel(Module):
            def __init__(self):
                super().__init__()
                self.conv = Conv2d(3, 8, kernel_size=3, padding=1, bias=True)
                self.bn = BatchNorm2d(8)
                self._modules["conv"] = self.conv
                self._modules["bn"] = self.bn

            def forward(self, x):
                # Use _modules dict so fusion updates are visible
                x = self._modules["conv"](x)
                x = self._modules["bn"](x)
                return x

        model = SimpleModel()
        model.train()

        # Run a few batches to build running stats
        for _ in range(5):
            x = np.random.randn(4, 3, 8, 8).astype(np.float32)
            model(x)

        model.eval()

        # Get unfused output
        x_test = np.random.randn(2, 3, 8, 8).astype(np.float32)
        out_unfused = model(x_test)

        # Fuse and get output
        converter = Converter()
        model_fused = converter.fuse_conv_bn(model)
        model_fused.eval()
        out_fused = model_fused(x_test)

        # Should be numerically close
        np.testing.assert_allclose(out_unfused, out_fused, atol=1e-4, rtol=1e-4)


class TestReLUReplacement:
    """Test ReLU -> IFNode replacement."""

    def test_replace_relu(self):
        """Converter should replace ReLU with IFNode."""
        from grilly.nn.module import Module
        from grilly.nn.modules import ReLU
        from grilly.nn.snn_ann2snn import Converter
        from grilly.nn.snn_neurons import IFNode

        class SimpleModel(Module):
            def __init__(self):
                super().__init__()
                self.relu = ReLU()
                self._modules["relu"] = self.relu

            def forward(self, x):
                return self.relu(x)

        model = SimpleModel()
        converter = Converter()
        model = converter.replace_relu_with_ifnode(model)

        assert isinstance(model._modules["relu"], IFNode)
        # IFNode should use soft reset for ANN2SNN
        assert model._modules["relu"].v_reset is None


class TestVoltageScaler:
    """Test VoltageScaler."""

    def test_scalar_scale(self):
        """VoltageScaler with scalar should scale uniformly."""
        from grilly.nn.snn_ann2snn import VoltageScaler

        scaler = VoltageScaler(scale=2.0)
        x = np.ones((2, 4), dtype=np.float32)
        out = scaler(x)
        np.testing.assert_allclose(out, 2.0)

    def test_per_channel_scale(self):
        """VoltageScaler with per-channel scale should work on 4D."""
        from grilly.nn.snn_ann2snn import VoltageScaler

        scale = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        scaler = VoltageScaler(scale=scale)
        x = np.ones((2, 3, 4, 4), dtype=np.float32)
        out = scaler(x)
        assert out[0, 0, 0, 0] == pytest.approx(1.0)
        assert out[0, 1, 0, 0] == pytest.approx(2.0)
        assert out[0, 2, 0, 0] == pytest.approx(3.0)


class TestFullConversion:
    """Test full ANN-to-SNN conversion pipeline."""

    def test_convert_simple_model(self):
        """Full conversion should produce a working SNN model."""
        from grilly.nn.module import Module
        from grilly.nn.modules import Linear, ReLU
        from grilly.nn.snn_ann2snn import Converter
        from grilly.nn.snn_neurons import IFNode

        class ANN(Module):
            def __init__(self):
                super().__init__()
                self.fc = Linear(16, 8, bias=True)
                self.relu = ReLU()
                self._modules["fc"] = self.fc
                self._modules["relu"] = self.relu

            def forward(self, x):
                return self.relu(self.fc(x))

        model = ANN()
        converter = Converter(mode="max", fuse_flag=False, T=4)
        snn = converter.convert(model)

        # ReLU should be replaced with IFNode
        assert isinstance(snn._modules["relu"], IFNode)

        # Should still be callable
        x = np.random.randn(2, 16).astype(np.float32)
        out = snn(x)
        assert out.shape == (2, 8) or out.shape == (2, 16)  # Depends on IFNode output
