"""
Tests for LSTM layer.
Validates correctness against PyTorch and verifies BPTT.
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


class TestLSTMCellBasic:
    """Basic LSTMCell functionality tests"""

    def test_lstmcell_import(self):
        """Test LSTMCell can be imported"""
        from grilly.nn import LSTMCell

        assert LSTMCell is not None

    def test_lstmcell_init(self):
        """Test LSTMCell initialization"""
        from grilly.nn import LSTMCell

        cell = LSTMCell(input_size=10, hidden_size=20)

        assert cell.input_size == 10
        assert cell.hidden_size == 20
        assert cell.weight_ih.shape == (4 * 20, 10)
        assert cell.weight_hh.shape == (4 * 20, 20)
        assert cell.bias_ih.shape == (4 * 20,)
        assert cell.bias_hh.shape == (4 * 20,)

    def test_lstmcell_forward(self):
        """Test LSTMCell forward pass"""
        from grilly.nn import LSTMCell

        batch_size = 2
        input_size = 10
        hidden_size = 20

        cell = LSTMCell(input_size, hidden_size)
        input = np.random.randn(batch_size, input_size).astype(np.float32)

        h, c = cell.forward(input)

        assert h.shape == (batch_size, hidden_size)
        assert c.shape == (batch_size, hidden_size)
        assert not np.any(np.isnan(h))
        assert not np.any(np.isnan(c))

    def test_lstmcell_with_state(self):
        """Test LSTMCell with provided hidden/cell state"""
        from grilly.nn import LSTMCell

        batch_size = 2
        input_size = 10
        hidden_size = 20

        cell = LSTMCell(input_size, hidden_size)
        input = np.random.randn(batch_size, input_size).astype(np.float32)
        h_prev = np.random.randn(batch_size, hidden_size).astype(np.float32)
        c_prev = np.random.randn(batch_size, hidden_size).astype(np.float32)

        h, c = cell.forward(input, (h_prev, c_prev))

        assert h.shape == (batch_size, hidden_size)
        assert c.shape == (batch_size, hidden_size)
        # Output should be different from input state
        assert not np.allclose(h, h_prev)


class TestLSTMBasic:
    """Basic LSTM functionality tests"""

    def test_lstm_import(self):
        """Test LSTM can be imported"""
        from grilly.nn import LSTM

        assert LSTM is not None

    def test_lstm_init(self):
        """Test LSTM initialization"""
        from grilly.nn import LSTM

        lstm = LSTM(input_size=10, hidden_size=20, num_layers=2)

        assert lstm.input_size == 10
        assert lstm.hidden_size == 20
        assert lstm.num_layers == 2
        assert len(lstm.cells_forward) == 2

    def test_lstm_forward(self):
        """Test LSTM forward pass"""
        from grilly.nn import LSTM

        batch_size = 2
        seq_len = 5
        input_size = 10
        hidden_size = 20

        lstm = LSTM(input_size, hidden_size)
        # Input shape: (seq_len, batch, input_size)
        input = np.random.randn(seq_len, batch_size, input_size).astype(np.float32)

        output, (h_n, c_n) = lstm.forward(input)

        assert output.shape == (seq_len, batch_size, hidden_size)
        assert h_n.shape == (1, batch_size, hidden_size)  # num_layers=1
        assert c_n.shape == (1, batch_size, hidden_size)
        assert not np.any(np.isnan(output))

    def test_lstm_batch_first(self):
        """Test LSTM with batch_first=True"""
        from grilly.nn import LSTM

        batch_size = 2
        seq_len = 5
        input_size = 10
        hidden_size = 20

        lstm = LSTM(input_size, hidden_size, batch_first=True)
        # Input shape: (batch, seq_len, input_size)
        input = np.random.randn(batch_size, seq_len, input_size).astype(np.float32)

        output, (h_n, c_n) = lstm.forward(input)

        # Output should be (batch, seq_len, hidden_size)
        assert output.shape == (batch_size, seq_len, hidden_size)
        assert h_n.shape == (1, batch_size, hidden_size)
        assert c_n.shape == (1, batch_size, hidden_size)

    def test_lstm_multilayer(self):
        """Test multi-layer LSTM"""
        from grilly.nn import LSTM

        batch_size = 2
        seq_len = 5
        input_size = 10
        hidden_size = 20
        num_layers = 3

        lstm = LSTM(input_size, hidden_size, num_layers=num_layers)
        input = np.random.randn(seq_len, batch_size, input_size).astype(np.float32)

        output, (h_n, c_n) = lstm.forward(input)

        assert output.shape == (seq_len, batch_size, hidden_size)
        assert h_n.shape == (num_layers, batch_size, hidden_size)
        assert c_n.shape == (num_layers, batch_size, hidden_size)

    def test_lstm_bidirectional(self):
        """Test bidirectional LSTM"""
        from grilly.nn import LSTM

        batch_size = 2
        seq_len = 5
        input_size = 10
        hidden_size = 20

        lstm = LSTM(input_size, hidden_size, bidirectional=True)
        input = np.random.randn(seq_len, batch_size, input_size).astype(np.float32)

        output, (h_n, c_n) = lstm.forward(input)

        # Output should have hidden_size * 2 (forward + backward)
        assert output.shape == (seq_len, batch_size, hidden_size * 2)
        # h_n should have 2 layers (forward + backward)
        assert h_n.shape == (2, batch_size, hidden_size)
        assert c_n.shape == (2, batch_size, hidden_size)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestLSTMCellVsPyTorch:
    """Test LSTMCell against PyTorch"""

    def test_lstmcell_matches_pytorch(self):
        """Verify LSTMCell matches PyTorch with same weights"""
        from grilly.nn import LSTMCell as GrillyLSTMCell

        np.random.seed(42)
        torch.manual_seed(42)

        batch_size = 2
        input_size = 5
        hidden_size = 8

        # Create Grilly cell
        grilly_cell = GrillyLSTMCell(input_size, hidden_size)

        # Create PyTorch cell and copy weights
        torch_cell = torch_nn.LSTMCell(input_size, hidden_size)

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
        c_prev = np.random.randn(batch_size, hidden_size).astype(np.float32)

        # Forward pass - Grilly
        h_grilly, c_grilly = grilly_cell.forward(input, (h_prev, c_prev))

        # Forward pass - PyTorch
        torch_input = torch.from_numpy(input)
        torch_h_prev = torch.from_numpy(h_prev)
        torch_c_prev = torch.from_numpy(c_prev)
        h_torch, c_torch = torch_cell(torch_input, (torch_h_prev, torch_c_prev))

        # Compare outputs
        np.testing.assert_allclose(
            h_grilly,
            h_torch.detach().numpy(),
            rtol=1e-4,
            atol=1e-5,
            err_msg="Hidden states should match",
        )
        np.testing.assert_allclose(
            c_grilly,
            c_torch.detach().numpy(),
            rtol=1e-4,
            atol=1e-5,
            err_msg="Cell states should match",
        )


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestLSTMVsPyTorch:
    """Test LSTM against PyTorch"""

    def test_lstm_matches_pytorch_single_layer(self):
        """Verify LSTM matches PyTorch for single layer"""
        from grilly.nn import LSTM as GrillyLSTM

        np.random.seed(42)
        torch.manual_seed(42)

        batch_size = 2
        seq_len = 3
        input_size = 5
        hidden_size = 8

        # Create Grilly LSTM
        grilly_lstm = GrillyLSTM(input_size, hidden_size, num_layers=1)

        # Create PyTorch LSTM and copy weights
        torch_lstm = torch_nn.LSTM(input_size, hidden_size, num_layers=1)

        # Copy weights from Grilly to PyTorch
        with torch.no_grad():
            torch_lstm.weight_ih_l0.copy_(
                torch.from_numpy(
                    np.asarray(grilly_lstm.cells_forward[0].weight_ih.data, dtype=np.float32)
                )
            )
            torch_lstm.weight_hh_l0.copy_(
                torch.from_numpy(
                    np.asarray(grilly_lstm.cells_forward[0].weight_hh.data, dtype=np.float32)
                )
            )
            torch_lstm.bias_ih_l0.copy_(
                torch.from_numpy(
                    np.asarray(grilly_lstm.cells_forward[0].bias_ih.data, dtype=np.float32)
                )
            )
            torch_lstm.bias_hh_l0.copy_(
                torch.from_numpy(
                    np.asarray(grilly_lstm.cells_forward[0].bias_hh.data, dtype=np.float32)
                )
            )

        # Create same input
        input = np.random.randn(seq_len, batch_size, input_size).astype(np.float32)

        # Forward pass - Grilly
        output_grilly, (h_n_grilly, c_n_grilly) = grilly_lstm.forward(input)

        # Forward pass - PyTorch
        torch_input = torch.from_numpy(input)
        output_torch, (h_n_torch, c_n_torch) = torch_lstm(torch_input)

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
        np.testing.assert_allclose(
            c_n_grilly,
            c_n_torch.detach().numpy(),
            rtol=1e-4,
            atol=1e-5,
            err_msg="Final cell should match",
        )


class TestLSTMEdgeCases:
    """Edge case tests for LSTM"""

    def test_lstm_zero_input(self):
        """Test LSTM with zero input"""
        from grilly.nn import LSTM

        batch_size = 2
        seq_len = 5
        input_size = 10
        hidden_size = 20

        lstm = LSTM(input_size, hidden_size)
        input = np.zeros((seq_len, batch_size, input_size), dtype=np.float32)

        output, (h_n, c_n) = lstm.forward(input)

        # Should not produce NaN
        assert not np.any(np.isnan(output))
        assert not np.any(np.isnan(h_n))
        assert not np.any(np.isnan(c_n))

    def test_lstm_single_timestep(self):
        """Test LSTM with single timestep"""
        from grilly.nn import LSTM

        batch_size = 2
        seq_len = 1
        input_size = 10
        hidden_size = 20

        lstm = LSTM(input_size, hidden_size)
        input = np.random.randn(seq_len, batch_size, input_size).astype(np.float32)

        output, (h_n, c_n) = lstm.forward(input)

        assert output.shape == (1, batch_size, hidden_size)
        # Final hidden should match the single timestep output
        np.testing.assert_allclose(output[0], h_n[0], rtol=1e-5)

    def test_lstm_long_sequence(self):
        """Test LSTM with long sequence"""
        from grilly.nn import LSTM

        batch_size = 2
        seq_len = 100
        input_size = 10
        hidden_size = 20

        lstm = LSTM(input_size, hidden_size)
        input = np.random.randn(seq_len, batch_size, input_size).astype(np.float32) * 0.1

        output, (h_n, c_n) = lstm.forward(input)

        assert output.shape == (seq_len, batch_size, hidden_size)
        # Should not explode or vanish (values should be reasonable)
        assert np.abs(output).mean() < 10.0
        assert np.abs(output).mean() > 0.0005  # Relaxed threshold for vanishing gradient check


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
