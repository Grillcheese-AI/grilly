#version 450

/*
LSTM Cell Forward Pass

Computes one timestep of LSTM:
  i_t = sigmoid(W_ii @ x_t + b_ii + W_hi @ h_{t-1} + b_hi)  # input gate
  f_t = sigmoid(W_if @ x_t + b_if + W_hf @ h_{t-1} + b_hf)  # forget gate
  g_t = tanh(W_ig @ x_t + b_ig + W_hg @ h_{t-1} + b_hg)     # cell gate
  o_t = sigmoid(W_io @ x_t + b_io + W_ho @ h_{t-1} + b_ho)  # output gate
  c_t = f_t * c_{t-1} + i_t * g_t                            # new cell state
  h_t = o_t * tanh(c_t)                                      # new hidden state

Input buffers:
  - input: (batch, input_size)
  - hidden: (batch, hidden_size) - h_{t-1}
  - cell: (batch, hidden_size) - c_{t-1}
  - weight_ih: (4 * hidden_size, input_size) - [W_ii, W_if, W_ig, W_io] stacked
  - weight_hh: (4 * hidden_size, hidden_size) - [W_hi, W_hf, W_hg, W_ho] stacked
  - bias_ih: (4 * hidden_size) - [b_ii, b_if, b_ig, b_io] stacked
  - bias_hh: (4 * hidden_size) - [b_hi, b_hf, b_hg, b_ho] stacked

Output buffers:
  - new_hidden: (batch, hidden_size) - h_t
  - new_cell: (batch, hidden_size) - c_t
  - gates: (batch, 4 * hidden_size) - [i, f, g, o] for backward pass
*/

layout(local_size_x = 256) in;

layout(push_constant) uniform PushConstants {
    uint batch_size;
    uint input_size;
    uint hidden_size;
} params;

layout(set = 0, binding = 0) readonly buffer InputBuffer {
    float input[];
};

layout(set = 0, binding = 1) readonly buffer HiddenBuffer {
    float hidden[];
};

layout(set = 0, binding = 2) readonly buffer CellBuffer {
    float cell[];
};

layout(set = 0, binding = 3) readonly buffer WeightIHBuffer {
    float weight_ih[];
};

layout(set = 0, binding = 4) readonly buffer WeightHHBuffer {
    float weight_hh[];
};

layout(set = 0, binding = 5) readonly buffer BiasIHBuffer {
    float bias_ih[];
};

layout(set = 0, binding = 6) readonly buffer BiasHHBuffer {
    float bias_hh[];
};

layout(set = 0, binding = 7) writeonly buffer NewHiddenBuffer {
    float new_hidden[];
};

layout(set = 0, binding = 8) writeonly buffer NewCellBuffer {
    float new_cell[];
};

layout(set = 0, binding = 9) writeonly buffer GatesBuffer {
    float gates[];
};

float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

void main() {
    uint global_id = gl_GlobalInvocationID.x;
    uint batch_idx = global_id / params.hidden_size;
    uint hidden_idx = global_id % params.hidden_size;

    if (batch_idx >= params.batch_size) return;

    // Compute all 4 gates (input, forget, cell, output)
    // Each gate is at offset gate_num * hidden_size
    for (uint gate = 0; gate < 4; gate++) {
        float value = 0.0;

        // Add W_i @ x_t
        for (uint i = 0; i < params.input_size; i++) {
            uint weight_idx = (gate * params.hidden_size + hidden_idx) * params.input_size + i;
            uint input_idx = batch_idx * params.input_size + i;
            value += weight_ih[weight_idx] * input[input_idx];
        }

        // Add bias_i
        value += bias_ih[gate * params.hidden_size + hidden_idx];

        // Add W_h @ h_{t-1}
        for (uint h = 0; h < params.hidden_size; h++) {
            uint weight_idx = (gate * params.hidden_size + hidden_idx) * params.hidden_size + h;
            uint hidden_prev_idx = batch_idx * params.hidden_size + h;
            value += weight_hh[weight_idx] * hidden[hidden_prev_idx];
        }

        // Add bias_h
        value += bias_hh[gate * params.hidden_size + hidden_idx];

        // Apply activation
        if (gate == 2) {
            // Cell gate uses tanh
            value = tanh(value);
        } else {
            // Input, forget, output gates use sigmoid
            value = sigmoid(value);
        }

        // Store gate value
        uint gate_idx = batch_idx * (4 * params.hidden_size) + gate * params.hidden_size + hidden_idx;
        gates[gate_idx] = value;
    }

    // Read gate values for this hidden unit
    uint gates_base = batch_idx * (4 * params.hidden_size) + hidden_idx;
    float i_gate = gates[gates_base];
    float f_gate = gates[gates_base + params.hidden_size];
    float g_gate = gates[gates_base + 2 * params.hidden_size];
    float o_gate = gates[gates_base + 3 * params.hidden_size];

    // Update cell state: c_t = f_t * c_{t-1} + i_t * g_t
    uint cell_idx = batch_idx * params.hidden_size + hidden_idx;
    float c_prev = cell[cell_idx];
    float c_new = f_gate * c_prev + i_gate * g_gate;
    new_cell[cell_idx] = c_new;

    // Update hidden state: h_t = o_t * tanh(c_t)
    float h_new = o_gate * tanh(c_new);
    new_hidden[cell_idx] = h_new;
}
