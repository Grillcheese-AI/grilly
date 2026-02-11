"""
Tests for GRU layer.
Validates correctness against PyTorch.
"""

import numpy as np
import pytest

# Try to import PyTorch for reference comparisons
try:
    import torch
    import torch.nn as torch_nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TestGRUCellBasic:
    """Basic GRUCell functionality tests"""

    def test_grucell_import(self):
        """Test GRUCell can be imported"""
        from grilly.nn import GRUCell

        assert GRUCell is not None

    def test_grucell_init(self):
        """Test GRUCell initialization"""
        from grilly.nn import GRUCell

        cell = GRUCell(input_size=10, hidden_size=20)

        assert cell.input_size == 10
        assert cell.hidden_size == 20
        assert cell.weight_ih.shape == (3 * 20, 10)  # 3 gates
        assert cell.weight_hh.shape == (3 * 20, 20)
        assert cell.bias_ih.shape == (3 * 20,)
        assert cell.bias_hh.shape == (3 * 20,)

    def test_grucell_forward(self):
        """Test GRUCell forward pass"""
        from grilly.nn import GRUCell

        batch_size = 2
        input_size = 10
        hidden_size = 20

        cell = GRUCell(input_size, hidden_size)
        input = np.random.randn(batch_size, input_size).astype(np.float32)

        h = cell.forward(input)

        assert h.shape == (batch_size, hidden_size)
        assert not np.any(np.isnan(h))

    def test_grucell_with_state(self):
        """Test GRUCell with provided hidden state"""
        from grilly.nn import GRUCell

        batch_size = 2
        input_size = 10
        hidden_size = 20

        cell = GRUCell(input_size, hidden_size)
        input = np.random.randn(batch_size, input_size).astype(np.float32)
        h_prev = np.random.randn(batch_size, hidden_size).astype(np.float32)

        h = cell.forward(input, h_prev)

        assert h.shape == (batch_size, hidden_size)
        # Output should be different from input state
        assert not np.allclose(h, h_prev)


class TestGRUBasic:
    """Basic GRU functionality tests"""

    def test_gru_import(self):
        """Test GRU can be imported"""
        from grilly.nn import GRU

        assert GRU is not None

    def test_gru_init(self):
        """Test GRU initialization"""
        from grilly.nn import GRU

        gru = GRU(input_size=10, hidden_size=20, num_layers=2)

        assert gru.input_size == 10
        assert gru.hidden_size == 20
        assert gru.num_layers == 2
        assert len(gru.cells_forward) == 2

    def test_gru_forward(self):
        """Test GRU forward pass"""
        from grilly.nn import GRU

        batch_size = 2
        seq_len = 5
        input_size = 10
        hidden_size = 20

        gru = GRU(input_size, hidden_size)
        # Input shape: (seq_len, batch, input_size)
        input = np.random.randn(seq_len, batch_size, input_size).astype(np.float32)

        output, h_n = gru.forward(input)

        assert output.shape == (seq_len, batch_size, hidden_size)
        assert h_n.shape == (1, batch_size, hidden_size)  # num_layers=1
        assert not np.any(np.isnan(output))

    def test_gru_batch_first(self):
        """Test GRU with batch_first=True"""
        from grilly.nn import GRU

        batch_size = 2
        seq_len = 5
        input_size = 10
        hidden_size = 20

        gru = GRU(input_size, hidden_size, batch_first=True)
        # Input shape: (batch, seq_len, input_size)
        input = np.random.randn(batch_size, seq_len, input_size).astype(np.float32)

        output, h_n = gru.forward(input)

        # Output should be (batch, seq_len, hidden_size)
        assert output.shape == (batch_size, seq_len, hidden_size)
        assert h_n.shape == (1, batch_size, hidden_size)

    def test_gru_multilayer(self):
        """Test multi-layer GRU"""
        from grilly.nn import GRU

        batch_size = 2
        seq_len = 5
        input_size = 10
        hidden_size = 20
        num_layers = 3

        gru = GRU(input_size, hidden_size, num_layers=num_layers)
        input = np.random.randn(seq_len, batch_size, input_size).astype(np.float32)

        output, h_n = gru.forward(input)

        assert output.shape == (seq_len, batch_size, hidden_size)
        assert h_n.shape == (num_layers, batch_size, hidden_size)

    def test_gru_bidirectional(self):
        """Test bidirectional GRU"""
        from grilly.nn import GRU

        batch_size = 2
        seq_len = 5
        input_size = 10
        hidden_size = 20

        gru = GRU(input_size, hidden_size, bidirectional=True)
        input = np.random.randn(seq_len, batch_size, input_size).astype(np.float32)

        output, h_n = gru.forward(input)

        # Output should have hidden_size * 2 (forward + backward)
        assert output.shape == (seq_len, batch_size, hidden_size * 2)
        # h_n should have 2 layers (forward + backward)
        assert h_n.shape == (2, batch_size, hidden_size)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestGRUCellVsPyTorch:
    """Test GRUCell against PyTorch"""

    def test_grucell_matches_pytorch(self):
        """Verify GRUCell matches PyTorch with same weights"""
        from grilly.nn import GRUCell as GrillyGRUCell

        np.random.seed(42)
        torch.manual_seed(42)

        batch_size = 2
        input_size = 5
        hidden_size = 8

        # Create Grilly cell
        grilly_cell = GrillyGRUCell(input_size, hidden_size)

        # Create PyTorch cell and copy weights
        torch_cell = torch_nn.GRUCell(input_size, hidden_size)

        # Copy weights from Grilly to PyTorch
        torch_cell.weight_ih.data = torch.from_numpy(
            np.asarray(grilly_cell.weight_ih.data, dtype=np.float32)
        )
        torch_cell.weight_hh.data = torch.from_numpy(
            np.asarray(grilly_cell.weight_hh.data, dtype=np.float32)
        )
        torch_cell.bias_ih.data = torch.from_numpy(
            np.asarray(grilly_cell.bias_ih.data, dtype=np.float32)
        )
        torch_cell.bias_hh.data = torch.from_numpy(
            np.asarray(grilly_cell.bias_hh.data, dtype=np.float32)
        )

        # Create same input
        input = np.random.randn(batch_size, input_size).astype(np.float32)
        h_prev = np.random.randn(batch_size, hidden_size).astype(np.float32)

        # Forward pass - Grilly
        h_grilly = grilly_cell.forward(input, h_prev)

        # Forward pass - PyTorch
        torch_input = torch.from_numpy(input)
        torch_h_prev = torch.from_numpy(h_prev)
        h_torch = torch_cell(torch_input, torch_h_prev)

        # Compare outputs
        np.testing.assert_allclose(
            h_grilly,
            h_torch.detach().numpy(),
            rtol=1e-4,
            atol=1e-5,
            err_msg="Hidden states should match",
        )


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestGRUVsPyTorch:
    """Test GRU against PyTorch"""

    def test_gru_matches_pytorch_single_layer(self):
        """Verify GRU matches PyTorch for single layer"""
        from grilly.nn import GRU as GrillyGRU

        np.random.seed(42)
        torch.manual_seed(42)

        batch_size = 2
        seq_len = 3
        input_size = 5
        hidden_size = 8

        # Create Grilly GRU
        grilly_gru = GrillyGRU(input_size, hidden_size, num_layers=1)

        # Create PyTorch GRU and copy weights
        torch_gru = torch_nn.GRU(input_size, hidden_size, num_layers=1)

        # Copy weights from Grilly to PyTorch
        with torch.no_grad():
            torch_gru.weight_ih_l0.copy_(
                torch.from_numpy(
                    np.asarray(grilly_gru.cells_forward[0].weight_ih.data, dtype=np.float32)
                )
            )
            torch_gru.weight_hh_l0.copy_(
                torch.from_numpy(
                    np.asarray(grilly_gru.cells_forward[0].weight_hh.data, dtype=np.float32)
                )
            )
            torch_gru.bias_ih_l0.copy_(
                torch.from_numpy(
                    np.asarray(grilly_gru.cells_forward[0].bias_ih.data, dtype=np.float32)
                )
            )
            torch_gru.bias_hh_l0.copy_(
                torch.from_numpy(
                    np.asarray(grilly_gru.cells_forward[0].bias_hh.data, dtype=np.float32)
                )
            )

        # Create same input
        input = np.random.randn(seq_len, batch_size, input_size).astype(np.float32)

        # Forward pass - Grilly
        output_grilly, h_n_grilly = grilly_gru.forward(input)

        # Forward pass - PyTorch
        torch_input = torch.from_numpy(input)
        output_torch, h_n_torch = torch_gru(torch_input)

        # Compare outputs
        np.testing.assert_allclose(
            output_grilly,
            output_torch.detach().numpy(),
            rtol=1e-4,
            atol=1e-5,
            err_msg="Outputs should match",
        )
        np.testing.assert_allclose(
            h_n_grilly,
            h_n_torch.detach().numpy(),
            rtol=1e-4,
            atol=1e-5,
            err_msg="Final hidden should match",
        )


class TestGRUEdgeCases:
    """Edge case tests for GRU"""

    def test_gru_zero_input(self):
        """Test GRU with zero input"""
        from grilly.nn import GRU

        batch_size = 2
        seq_len = 5
        input_size = 10
        hidden_size = 20

        gru = GRU(input_size, hidden_size)
        input = np.zeros((seq_len, batch_size, input_size), dtype=np.float32)

        output, h_n = gru.forward(input)

        # Should not produce NaN
        assert not np.any(np.isnan(output))
        assert not np.any(np.isnan(h_n))

    def test_gru_single_timestep(self):
        """Test GRU with single timestep"""
        from grilly.nn import GRU

        batch_size = 2
        seq_len = 1
        input_size = 10
        hidden_size = 20

        gru = GRU(input_size, hidden_size)
        input = np.random.randn(seq_len, batch_size, input_size).astype(np.float32)

        output, h_n = gru.forward(input)

        assert output.shape == (1, batch_size, hidden_size)
        # Final hidden should match the single timestep output
        np.testing.assert_allclose(output[0], h_n[0], rtol=1e-5)

    def test_gru_long_sequence(self):
        """Test GRU with long sequence"""
        from grilly.nn import GRU

        batch_size = 2
        seq_len = 100
        input_size = 10
        hidden_size = 20

        gru = GRU(input_size, hidden_size)
        input = np.random.randn(seq_len, batch_size, input_size).astype(np.float32) * 0.1

        output, h_n = gru.forward(input)

        assert output.shape == (seq_len, batch_size, hidden_size)
        # Should not explode or vanish (values should be reasonable)
        assert np.abs(output).mean() < 10.0
        assert np.abs(output).mean() > 0.001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
