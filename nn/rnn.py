"""Recurrent neural network modules for LSTM and GRU architectures."""

import numpy as np

from .module import Module
from .parameter import Parameter


class LSTMCell(Module):
    """Long short-term memory cell compatible with torch.nn.LSTMCell."""

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        """Initialize an LSTM cell with input and hidden dimensions."""
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        # Weight matrices: [input, forget, gate, output] stacked
        # Input-to-hidden weights: (4 * hidden_size, input_size)
        self.weight_ih = Parameter(
            np.random.randn(4 * hidden_size, input_size).astype(np.float32) * 0.01
        )

        # Hidden-to-hidden weights: (4 * hidden_size, hidden_size)
        self.weight_hh = Parameter(
            np.random.randn(4 * hidden_size, hidden_size).astype(np.float32) * 0.01
        )

        if bias:
            # Bias terms: (4 * hidden_size)
            self.bias_ih = Parameter(np.zeros(4 * hidden_size, dtype=np.float32))
            self.bias_hh = Parameter(np.zeros(4 * hidden_size, dtype=np.float32))
        else:
            self.bias_ih = None
            self.bias_hh = None

    def forward(
        self, input: np.ndarray, hx: tuple[np.ndarray, np.ndarray] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute one LSTM step and return ``(h_new, c_new)``."""
        batch_size = input.shape[0]

        # Initialize hidden and cell states if not provided
        if hx is None:
            h_prev = np.zeros((batch_size, self.hidden_size), dtype=np.float32)
            c_prev = np.zeros((batch_size, self.hidden_size), dtype=np.float32)
        else:
            h_prev, c_prev = hx

        # Ensure inputs are numpy arrays
        if not isinstance(input, np.ndarray):
            input = np.asarray(input, dtype=np.float32)
        if not isinstance(h_prev, np.ndarray):
            h_prev = np.asarray(h_prev, dtype=np.float32)
        if not isinstance(c_prev, np.ndarray):
            c_prev = np.asarray(c_prev, dtype=np.float32)

        # Ensure weights are numpy arrays (handle memoryview)
        weight_ih = np.asarray(self.weight_ih.data, dtype=np.float32)
        weight_hh = np.asarray(self.weight_hh.data, dtype=np.float32)

        # Compute gates: W_ih @ x + b_ih + W_hh @ h + b_hh
        # Shape: (batch, 4 * hidden_size)
        gates = input @ weight_ih.T + h_prev @ weight_hh.T

        if self.bias:
            bias_ih = np.asarray(self.bias_ih.data, dtype=np.float32)
            bias_hh = np.asarray(self.bias_hh.data, dtype=np.float32)
            gates += bias_ih + bias_hh

        # Split into 4 gates: input, forget, cell, output
        # Each gate has shape (batch, hidden_size)
        chunk_size = self.hidden_size
        i_gate = self._sigmoid(gates[:, 0 * chunk_size : 1 * chunk_size])
        f_gate = self._sigmoid(gates[:, 1 * chunk_size : 2 * chunk_size])
        g_gate = np.tanh(gates[:, 2 * chunk_size : 3 * chunk_size])
        o_gate = self._sigmoid(gates[:, 3 * chunk_size : 4 * chunk_size])

        # Update cell state: c_t = f_t * c_{t-1} + i_t * g_t
        c_new = f_gate * c_prev + i_gate * g_gate

        # Update hidden state: h_t = o_t * tanh(c_t)
        h_new = o_gate * np.tanh(c_new)

        return h_new, c_new

    def _sigmoid(self, x):
        """Sigmoid activation function."""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


class LSTM(Module):
    """
    Long Short-Term Memory (LSTM) network.

    Matches torch.nn.LSTM

    Applies a multi-layer LSTM to an input sequence.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        """
        Initialize LSTM.

        Args:
            input_size: Number of expected features in input
            hidden_size: Number of features in hidden state
            num_layers: Number of recurrent layers (default: 1)
            bias: If False, layer does not use bias weights (default: True)
            batch_first: If True, input/output tensors are (batch, seq, feature)
                        else (seq, batch, feature) (default: False)
            dropout: Dropout probability for outputs of each layer except last (default: 0)
            bidirectional: If True, becomes bidirectional LSTM (default: False)
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Create LSTM cells for each layer and direction
        self.cells_forward = []
        self.cells_backward = []

        for layer in range(num_layers):
            # First layer takes input_size, subsequent layers take hidden_size
            layer_input_size = input_size if layer == 0 else hidden_size * self.num_directions

            # Forward direction
            self.cells_forward.append(LSTMCell(layer_input_size, hidden_size, bias))

            # Backward direction (if bidirectional)
            if bidirectional:
                self.cells_backward.append(LSTMCell(layer_input_size, hidden_size, bias))

    def forward(
        self, input: np.ndarray, hx: tuple[np.ndarray, np.ndarray] | None = None
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """
        Forward pass through LSTM.

        Args:
            input: Input tensor of shape:
                   - (seq_len, batch, input_size) if batch_first=False
                   - (batch, seq_len, input_size) if batch_first=True
            hx: Tuple of (h_0, c_0) where:
                h_0: Initial hidden state (num_layers * num_directions, batch, hidden_size)
                c_0: Initial cell state (num_layers * num_directions, batch, hidden_size)
                If None, initialized to zeros

        Returns:
            Tuple of (output, (h_n, c_n)) where:
                output: Tensor of shape:
                        - (seq_len, batch, hidden_size * num_directions) if batch_first=False
                        - (batch, seq_len, hidden_size * num_directions) if batch_first=True
                h_n: Final hidden state (num_layers * num_directions, batch, hidden_size)
                c_n: Final cell state (num_layers * num_directions, batch, hidden_size)
        """
        # Handle batch_first
        if self.batch_first:
            input = np.swapaxes(input, 0, 1)  # (batch, seq, features) -> (seq, batch, features)

        seq_len, batch_size, _ = input.shape

        # Initialize hidden and cell states if not provided
        if hx is None:
            h_0 = np.zeros(
                (self.num_layers * self.num_directions, batch_size, self.hidden_size),
                dtype=np.float32,
            )
            c_0 = np.zeros(
                (self.num_layers * self.num_directions, batch_size, self.hidden_size),
                dtype=np.float32,
            )
        else:
            h_0, c_0 = hx

        # Process each layer
        layer_output = input
        final_hiddens = []
        final_cells = []

        for layer in range(self.num_layers):
            # Get initial states for this layer
            h_forward = h_0[layer * self.num_directions]
            c_forward = c_0[layer * self.num_directions]

            # Forward direction
            forward_outputs = []
            h_t, c_t = h_forward, c_forward

            for t in range(seq_len):
                h_t, c_t = self.cells_forward[layer](layer_output[t], (h_t, c_t))
                forward_outputs.append(h_t)

            forward_outputs = np.stack(forward_outputs, axis=0)  # (seq_len, batch, hidden_size)
            final_hiddens.append(h_t)
            final_cells.append(c_t)

            # Backward direction (if bidirectional)
            if self.bidirectional:
                h_backward = h_0[layer * self.num_directions + 1]
                c_backward = c_0[layer * self.num_directions + 1]

                backward_outputs = []
                h_t, c_t = h_backward, c_backward

                for t in range(seq_len - 1, -1, -1):
                    h_t, c_t = self.cells_backward[layer](layer_output[t], (h_t, c_t))
                    backward_outputs.insert(0, h_t)

                backward_outputs = np.stack(
                    backward_outputs, axis=0
                )  # (seq_len, batch, hidden_size)
                final_hiddens.append(h_t)
                final_cells.append(c_t)

                # Concatenate forward and backward outputs
                layer_output = np.concatenate([forward_outputs, backward_outputs], axis=2)
            else:
                layer_output = forward_outputs

            # Apply dropout (except on last layer)
            if layer < self.num_layers - 1 and self.dropout > 0:
                mask = np.random.binomial(1, 1 - self.dropout, layer_output.shape)
                layer_output = layer_output * mask / (1 - self.dropout)

        # Stack final hidden and cell states
        h_n = np.stack(final_hiddens, axis=0)  # (num_layers * num_directions, batch, hidden_size)
        c_n = np.stack(final_cells, axis=0)

        # Handle batch_first for output
        output = layer_output
        if self.batch_first:
            output = np.swapaxes(output, 0, 1)  # (seq, batch, features) -> (batch, seq, features)

        return output, (h_n, c_n)


class GRUCell(Module):
    """
    Gated Recurrent Unit (GRU) cell.

    Matches torch.nn.GRUCell

    Applies GRU transformation:
    r_t = sigmoid(W_ir @ x_t + b_ir + W_hr @ h_{t-1} + b_hr)
    z_t = sigmoid(W_iz @ x_t + b_iz + W_hz @ h_{t-1} + b_hz)
    n_t = tanh(W_in @ x_t + b_in + r_t * (W_hn @ h_{t-1} + b_hn))
    h_t = (1 - z_t) * n_t + z_t * h_{t-1}
    """

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        """
        Initialize GRUCell.

        Args:
            input_size: Number of expected features in input
            hidden_size: Number of features in hidden state
            bias: If False, layer does not use bias weights (default: True)
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        # Weight matrices: [reset, update, new] stacked
        # Input-to-hidden weights: (3 * hidden_size, input_size)
        self.weight_ih = Parameter(
            np.random.randn(3 * hidden_size, input_size).astype(np.float32) * 0.01
        )

        # Hidden-to-hidden weights: (3 * hidden_size, hidden_size)
        self.weight_hh = Parameter(
            np.random.randn(3 * hidden_size, hidden_size).astype(np.float32) * 0.01
        )

        if bias:
            # Bias terms: (3 * hidden_size)
            self.bias_ih = Parameter(np.zeros(3 * hidden_size, dtype=np.float32))
            self.bias_hh = Parameter(np.zeros(3 * hidden_size, dtype=np.float32))
        else:
            self.bias_ih = None
            self.bias_hh = None

    def forward(self, input: np.ndarray, hx: np.ndarray | None = None) -> np.ndarray:
        """
        Forward pass through GRU cell.

        Args:
            input: Input tensor of shape (batch, input_size)
            hx: Initial hidden state (batch, hidden_size). If None, initialized to zeros

        Returns:
            h_1: Next hidden state (batch, hidden_size)
        """
        batch_size = input.shape[0]

        # Initialize hidden state if not provided
        if hx is None:
            h_prev = np.zeros((batch_size, self.hidden_size), dtype=np.float32)
        else:
            h_prev = hx

        # Ensure inputs are numpy arrays
        if not isinstance(input, np.ndarray):
            input = np.asarray(input, dtype=np.float32)
        if not isinstance(h_prev, np.ndarray):
            h_prev = np.asarray(h_prev, dtype=np.float32)

        # Ensure weights are numpy arrays (handle memoryview)
        weight_ih = np.asarray(self.weight_ih.data, dtype=np.float32)
        weight_hh = np.asarray(self.weight_hh.data, dtype=np.float32)

        # Compute input and hidden transformations
        gi = input @ weight_ih.T  # (batch, 3 * hidden_size)
        gh = h_prev @ weight_hh.T  # (batch, 3 * hidden_size)

        if self.bias:
            bias_ih = np.asarray(self.bias_ih.data, dtype=np.float32)
            bias_hh = np.asarray(self.bias_hh.data, dtype=np.float32)
            gi += bias_ih
            gh += bias_hh

        # Split into 3 gates: reset, update, new
        i_reset, i_update, i_new = np.split(gi, 3, axis=1)
        h_reset, h_update, h_new = np.split(gh, 3, axis=1)

        # Reset gate: r_t = sigmoid(W_ir @ x + W_hr @ h)
        r_gate = self._sigmoid(i_reset + h_reset)

        # Update gate: z_t = sigmoid(W_iz @ x + W_hz @ h)
        z_gate = self._sigmoid(i_update + h_update)

        # New gate: n_t = tanh(W_in @ x + r * (W_hn @ h))
        n_gate = np.tanh(i_new + r_gate * h_new)

        # New hidden: h_t = (1 - z) * n + z * h
        h_new = (1 - z_gate) * n_gate + z_gate * h_prev

        return h_new

    def _sigmoid(self, x):
        """Sigmoid activation function."""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


class GRU(Module):
    """
    Gated Recurrent Unit (GRU) network.

    Matches torch.nn.GRU

    Applies a multi-layer GRU to an input sequence.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        """
        Initialize GRU.

        Args:
            input_size: Number of expected features in input
            hidden_size: Number of features in hidden state
            num_layers: Number of recurrent layers (default: 1)
            bias: If False, layer does not use bias weights (default: True)
            batch_first: If True, input/output tensors are (batch, seq, feature) (default: False)
            dropout: Dropout probability for outputs of each layer except last (default: 0)
            bidirectional: If True, becomes bidirectional GRU (default: False)
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Create GRU cells for each layer and direction
        self.cells_forward = []
        self.cells_backward = []

        for layer in range(num_layers):
            # First layer takes input_size, subsequent layers take hidden_size
            layer_input_size = input_size if layer == 0 else hidden_size * self.num_directions

            # Forward direction
            self.cells_forward.append(GRUCell(layer_input_size, hidden_size, bias))

            # Backward direction (if bidirectional)
            if bidirectional:
                self.cells_backward.append(GRUCell(layer_input_size, hidden_size, bias))

    def forward(
        self, input: np.ndarray, hx: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through GRU.

        Args:
            input: Input tensor of shape:
                   - (seq_len, batch, input_size) if batch_first=False
                   - (batch, seq_len, input_size) if batch_first=True
            hx: Initial hidden state (num_layers * num_directions, batch, hidden_size)
                If None, initialized to zeros

        Returns:
            Tuple of (output, h_n) where:
                output: Tensor of shape:
                        - (seq_len, batch, hidden_size * num_directions) if batch_first=False
                        - (batch, seq_len, hidden_size * num_directions) if batch_first=True
                h_n: Final hidden state (num_layers * num_directions, batch, hidden_size)
        """
        # Handle batch_first
        if self.batch_first:
            input = np.swapaxes(input, 0, 1)  # (batch, seq, features) -> (seq, batch, features)

        seq_len, batch_size, _ = input.shape

        # Initialize hidden state if not provided
        if hx is None:
            h_0 = np.zeros(
                (self.num_layers * self.num_directions, batch_size, self.hidden_size),
                dtype=np.float32,
            )
        else:
            h_0 = hx

        # Process each layer
        layer_output = input
        final_hiddens = []

        for layer in range(self.num_layers):
            # Get initial state for this layer
            h_forward = h_0[layer * self.num_directions]

            # Forward direction
            forward_outputs = []
            h_t = h_forward

            for t in range(seq_len):
                h_t = self.cells_forward[layer](layer_output[t], h_t)
                forward_outputs.append(h_t)

            forward_outputs = np.stack(forward_outputs, axis=0)  # (seq_len, batch, hidden_size)
            final_hiddens.append(h_t)

            # Backward direction (if bidirectional)
            if self.bidirectional:
                h_backward = h_0[layer * self.num_directions + 1]

                backward_outputs = []
                h_t = h_backward

                for t in range(seq_len - 1, -1, -1):
                    h_t = self.cells_backward[layer](layer_output[t], h_t)
                    backward_outputs.insert(0, h_t)

                backward_outputs = np.stack(
                    backward_outputs, axis=0
                )  # (seq_len, batch, hidden_size)
                final_hiddens.append(h_t)

                # Concatenate forward and backward outputs
                layer_output = np.concatenate([forward_outputs, backward_outputs], axis=2)
            else:
                layer_output = forward_outputs

            # Apply dropout (except on last layer)
            if layer < self.num_layers - 1 and self.dropout > 0:
                mask = np.random.binomial(1, 1 - self.dropout, layer_output.shape)
                layer_output = layer_output * mask / (1 - self.dropout)

        # Stack final hidden states
        h_n = np.stack(final_hiddens, axis=0)  # (num_layers * num_directions, batch, hidden_size)

        # Handle batch_first for output
        output = layer_output
        if self.batch_first:
            output = np.swapaxes(output, 0, 1)  # (seq, batch, features) -> (batch, seq, features)

        return output, h_n
