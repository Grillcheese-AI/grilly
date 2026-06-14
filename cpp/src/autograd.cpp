#include "grilly/autograd/autograd.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "grilly/ops/linear.h"

namespace grilly {
namespace autograd {

// ═════════════════════════════════════════════════════════════════════════
// BackwardEngine
// ═════════════════════════════════════════════════════════════════════════

BackwardEngine::BackwardEngine(BufferPool& pool, CommandBatch& batch,
                               PipelineCache& cache, BufferRegistry& registry)
    : pool_(pool), batch_(batch), cache_(cache), registry_(registry) {
    clear_grads();
}

void BackwardEngine::backward(TapeArena& tape, Node* loss_node,
                               uint32_t grad_output_buffer) {
    stats_ = {};

    // 1. Seed the loss node's gradient: dL/d(loss) flows in from outside
    loss_node->grad_output_buffer = grad_output_buffer;

    // Open a single command batch. Every backward handler records its shader
    // dispatches into batch_ without submitting; the whole backward pass is
    // one submit at the end (resident — grad buffers never leave VRAM).
    batch_.begin();

    // 2. Walk the Wengert list backward: tail → prev → prev → nullptr
    //
    //    Because the allocation order is a valid topological order,
    //    walking backward guarantees that when we process node N,
    //    all nodes that USE node N's output (i.e., nodes allocated AFTER N)
    //    have already been processed and have accumulated their gradient
    //    contributions into N's grad_output_buffer.
    Node* current = tape.tail();

    while (current != nullptr) {
        stats_.nodes_visited++;

        if (current->grad_output_buffer != 0) {
            stats_.nodes_with_grad++;

            // Dispatch the backward shader for this operation
            dispatch_node_backward(current);

            // Accumulate gradients into input nodes.
            // For each input that requires_grad, find/create a grad buffer
            // and add this node's contribution to it (handles fan-out).
            for (uint32_t i = 0; i < current->num_inputs; ++i) {
                if (!current->inputs[i].requires_grad) continue;
                if (current->grad_input_buffers[i] == 0) continue;

                uint32_t input_buf = current->inputs[i].buffer_id;
                uint32_t& accum = find_or_insert_grad(input_buf);

                if (accum == 0) {
                    // First gradient contribution — just store it
                    accum = current->grad_input_buffers[i];
                } else {
                    // Fan-out: add this gradient to the accumulated one.
                    // Dispatch an element-wise add shader: accum += new_grad
                    //
                    // We use the existing "activation-add" shader which does:
                    //   output[i] = input_a[i] + input_b[i]
                    // Here we read from accum + new_grad, write back to accum.
                    size_t grad_size = current->inputs[i].size_bytes();
                    GrillyBuffer accum_buf = pool_.acquire(grad_size);
                    GrillyBuffer new_grad_buf = pool_.acquire(grad_size);

                    // For now, mark that accumulation happened.
                    // The actual Vulkan add dispatch is wired in Phase 2
                    // when we connect specific backward shaders.
                    // TODO: dispatch element-wise add shader
                    (void)accum_buf;
                    (void)new_grad_buf;
                }

                // Propagate: set the input node's grad_output_buffer so
                // when we reach that node in the walk, it knows to run.
                // We need to find the node that PRODUCED this input.
                // In the Wengert list, that node was allocated earlier.
                // We propagate by storing the grad in the grad_table.
            }
        }

        current = current->prev_in_tape;
    }

    // Submit the whole backward pass as one batch and wait for completion,
    // so gradient buffers are valid for readback / the optimizer step.
    batch_.submit();
}

void BackwardEngine::dispatch_node_backward(Node* node) {
    // Dispatch table — select the backward implementation based on OpType.
    // Each handler reads node->grad_output_buffer and writes
    // node->grad_input_buffers[i] for each input that requires_grad.
    switch (node->op) {
        case OpType::Linear:    backward_linear(node); break;
        case OpType::MatMul:    backward_matmul(node); break;
        case OpType::ReLU:      backward_relu(node); break;
        case OpType::GELU:      backward_gelu(node); break;
        case OpType::SiLU:      backward_silu(node); break;
        case OpType::Tanh:      backward_tanh(node); break;
        case OpType::Sigmoid:   backward_sigmoid(node); break;
        case OpType::Softmax:   backward_softmax(node); break;
        case OpType::LayerNorm: backward_layernorm(node); break;
        case OpType::FlashAttention2: backward_attention(node); break;
        case OpType::Conv2d:    backward_conv2d(node); break;
        case OpType::Conv1d:    backward_conv1d(node); break;
        case OpType::Add:       backward_add(node); break;
        case OpType::Sub:       backward_sub(node); break;
        case OpType::Mul:       backward_mul(node); break;
        case OpType::Div:       backward_div(node); break;
        case OpType::CrossEntropy: backward_cross_entropy(node); break;
        case OpType::MSELoss:   backward_mse(node); break;
        case OpType::CubeMindSurprise: backward_cubemind_surprise(node); break;
        case OpType::TemporalSurprise: backward_temporal_surprise(node); break;
        case OpType::Reshape:   backward_reshape(node); break;
        case OpType::Transpose: backward_transpose(node); break;
        case OpType::Sum:       backward_sum(node); break;
        case OpType::Mean:      backward_mean(node); break;

        // Ops with no backward (or not yet implemented)
        default:
            stats_.cpu_fallbacks++;
            break;
    }
}

uint32_t BackwardEngine::get_grad_buffer(uint32_t input_buffer_id) const {
    for (uint32_t i = 0; i < grad_count_; ++i) {
        if (grad_table_[i].buffer_id == input_buffer_id) {
            return grad_table_[i].grad_buffer_id;
        }
    }
    return 0;
}

void BackwardEngine::clear_grads() {
    grad_count_ = 0;
    std::memset(grad_table_, 0, sizeof(grad_table_));
}

uint32_t& BackwardEngine::find_or_insert_grad(uint32_t buffer_id) {
    // Linear scan — fine for <4096 entries. The grad_table_ is in L1 cache
    // because it's a contiguous array on the BackwardEngine (stack/heap).
    for (uint32_t i = 0; i < grad_count_; ++i) {
        if (grad_table_[i].buffer_id == buffer_id) {
            return grad_table_[i].grad_buffer_id;
        }
    }
    // Insert new entry
    if (grad_count_ < kMaxGradEntries) {
        grad_table_[grad_count_].buffer_id = buffer_id;
        grad_table_[grad_count_].grad_buffer_id = 0;
        return grad_table_[grad_count_++].grad_buffer_id;
    }
    // Overflow — should not happen in practice
    overflow_slot_ = 0;
    return overflow_slot_;
}

void BackwardEngine::accumulate_grad(Node* target_node, uint32_t input_idx,
                                      uint32_t grad_buffer) {
    // Store the gradient buffer in the target node's grad_input_buffers
    if (input_idx < kMaxNodeIO) {
        target_node->grad_input_buffers[input_idx] = grad_buffer;
    }
}

// ═════════════════════════════════════════════════════════════════════════
// Backward Shader Dispatch Implementations
// ═════════════════════════════════════════════════════════════════════════
//
// Each handler follows the same pattern:
//   1. Read grad_output from node->grad_output_buffer
//   2. Read saved tensors from node->saved_buffer_ids[]
//   3. Allocate output grad buffers from BufferPool
//   4. Dispatch the appropriate Vulkan compute shader
//   5. Store result buffer IDs in node->grad_input_buffers[]
//
// For Phase 1, we implement the shader dispatch scaffolding.
// The actual VkDescriptorBufferInfo setup and shader names match
// the existing GLSL shaders in shaders/spv/.

void BackwardEngine::backward_linear(Node* node) {
    // Linear: y = x @ W^T (+ b). W has shape (outputDim, inputDim).
    // Backward (all via the fnn-linear-backward shader, 3 passes):
    //   pass 0: dL/dx = dL/dy @ W
    //   pass 1: dL/dW = dL/dy^T @ x   (atomic accumulate -> needs zero init)
    //   pass 2: dL/db = sum(dL/dy, dim=0) (atomic accumulate -> needs zero init)
    //
    // inputs[0] = x, inputs[1] = W, (optional) inputs[2] = b
    // saved_buffer_ids[0] = x, saved_buffer_ids[1] = W
    // grad_output_buffer holds dL/dy (resident).

    // Shapes. W is (outputDim, inputDim); x is (..., inputDim) flattened to
    // (batchSeq, inputDim). Derive dims from the weight ref (unambiguous) and
    // batchSeq from x's element count.
    const TensorRef& xRef = node->inputs[0];
    const TensorRef& wRef = node->inputs[1];
    if (wRef.ndim < 2) {
        stats_.cpu_fallbacks++;
        return;
    }
    const uint32_t outputDim = wRef.shape[0];
    const uint32_t inputDim  = wRef.shape[1];
    if (inputDim == 0) { stats_.cpu_fallbacks++; return; }
    const uint32_t batchSeq =
        static_cast<uint32_t>(xRef.numel() / inputDim);

    // Resolve resident inputs from the registry.
    GrillyBuffer& gradOut = registry_.resolve(node->grad_output_buffer);
    // x and W come from saved buffers (fall back to input refs if unsaved).
    uint32_t xId = (node->num_saved > 0 && node->saved_buffer_ids[0] != 0)
                       ? node->saved_buffer_ids[0] : xRef.buffer_id;
    uint32_t wId = (node->num_saved > 1 && node->saved_buffer_ids[1] != 0)
                       ? node->saved_buffer_ids[1] : wRef.buffer_id;
    GrillyBuffer& xBuf = registry_.resolve(xId);
    GrillyBuffer& wBuf = registry_.resolve(wId);

    // Byte sizes (fp32).
    const size_t gradOutBytes = size_t(batchSeq) * outputDim * 4;
    const size_t gradInBytes  = size_t(batchSeq) * inputDim * 4;
    const size_t gradWBytes   = size_t(outputDim) * inputDim * 4;
    const size_t gradBiasBytes = size_t(outputDim) * 4;

    // Allocate resident grad buffers (DEVICE_LOCAL) and record their real ids.
    // The shader writes all three bindings every pass, so all three must
    // exist even if a given input doesn't requires_grad; we allocate them and
    // only publish ids for inputs that need them.
    uint32_t gradInId = registry_.alloc(gradInBytes);
    uint32_t gradWId  = registry_.alloc(gradWBytes);
    uint32_t gradBiasId = registry_.alloc(gradBiasBytes);
    GrillyBuffer& gradInBuf = registry_.resolve(gradInId);
    GrillyBuffer& gradWBuf  = registry_.resolve(gradWId);
    GrillyBuffer& gradBiasBuf = registry_.resolve(gradBiasId);

    // Descriptor set: 6 buffers in the order the shader expects.
    PipelineEntry pipe = cache_.getOrCreate("fnn-linear-backward", 6,
                                             sizeof(grilly::ops::LinearBackwardParams));
    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {gradOut.handle,     0, gradOutBytes},
        {xBuf.handle,        0, size_t(batchSeq) * inputDim * 4},
        {wBuf.handle,        0, gradWBytes},
        {gradInBuf.handle,   0, gradInBytes},
        {gradWBuf.handle,    0, gradWBytes},
        {gradBiasBuf.handle, 0, gradBiasBytes},
    };
    VkDescriptorSet descSet =
        cache_.allocDescriptorSet("fnn-linear-backward", bufInfos);

    // Zero the accumulation targets GPU-side (passes 1 and 2 atomic-add).
    // grad_input (pass 0) is fully written, but zero it too for safety.
    batch_.fillZero(gradInBuf, gradInBytes);
    batch_.fillZero(gradWBuf, gradWBytes);
    batch_.fillZero(gradBiasBuf, gradBiasBytes);
    batch_.transferComputeBarrier();

    grilly::ops::LinearBackwardParams p{batchSeq, inputDim, outputDim, 0};

    // Pass 0: grad_input = grad_output @ W
    p.passType = 0;
    batch_.dispatch(pipe.pipeline, pipe.layout, descSet,
                    (inputDim + 15) / 16, (batchSeq + 15) / 16, 1,
                    &p, sizeof(p));
    batch_.barrier();

    // Pass 1: grad_weight = grad_output^T @ x
    p.passType = 1;
    batch_.dispatch(pipe.pipeline, pipe.layout, descSet,
                    (inputDim + 15) / 16, (outputDim + 15) / 16, 1,
                    &p, sizeof(p));
    batch_.barrier();

    // Pass 2: grad_bias = sum(grad_output, dim=0)
    p.passType = 2;
    batch_.dispatch(pipe.pipeline, pipe.layout, descSet,
                    (outputDim + 255) / 256, 1, 1,
                    &p, sizeof(p));

    stats_.shaders_dispatched++;

    // Publish gradient ids for inputs that require grad. Buffers for inputs
    // that don't require grad were still allocated (shader writes them) and
    // will be released by registry_.clear() at the next begin().
    if (xRef.requires_grad) node->grad_input_buffers[0] = gradInId;
    if (node->num_inputs > 1 && wRef.requires_grad)
        node->grad_input_buffers[1] = gradWId;
    if (node->num_inputs > 2 && node->inputs[2].requires_grad)
        node->grad_input_buffers[2] = gradBiasId;
}

void BackwardEngine::backward_matmul(Node* node) {
    // MatMul: C = A @ B
    // dL/dA = dL/dC @ B^T
    // dL/dB = A^T @ dL/dC
    stats_.shaders_dispatched++;

    if (node->inputs[0].requires_grad) {
        node->grad_input_buffers[0] = 1;  // placeholder
        // TODO: dispatch matmul shader with transposed B
    }
    if (node->inputs[1].requires_grad) {
        node->grad_input_buffers[1] = 1;  // placeholder
        // TODO: dispatch matmul shader with transposed A
    }
}

void BackwardEngine::backward_relu(Node* node) {
    // ReLU: y = max(0, x)
    // dL/dx = dL/dy * (x > 0)
    // Uses saved pre-activation x to compute the mask.
    stats_.shaders_dispatched++;

    if (node->inputs[0].requires_grad) {
        node->grad_input_buffers[0] = 1;  // placeholder
        // TODO: dispatch activation-relu-backward.glsl
        //   binding 0: grad_output
        //   binding 1: saved input (x)
        //   binding 2: grad_input (output)
    }
}

void BackwardEngine::backward_gelu(Node* node) {
    // GELU backward — uses the pre-activation value
    stats_.shaders_dispatched++;
    if (node->inputs[0].requires_grad) {
        node->grad_input_buffers[0] = 1;
        // TODO: dispatch activation-gelu-backward.glsl
    }
}

void BackwardEngine::backward_silu(Node* node) {
    stats_.shaders_dispatched++;
    if (node->inputs[0].requires_grad) {
        node->grad_input_buffers[0] = 1;
        // TODO: dispatch activation-silu-backward.glsl
    }
}

void BackwardEngine::backward_tanh(Node* node) {
    // tanh backward: dL/dx = dL/dy * (1 - tanh(x)^2)
    // We save the output y = tanh(x) and compute 1 - y^2
    stats_.shaders_dispatched++;
    if (node->inputs[0].requires_grad) {
        node->grad_input_buffers[0] = 1;
        // TODO: dispatch tanh backward shader
    }
}

void BackwardEngine::backward_sigmoid(Node* node) {
    // sigmoid backward: dL/dx = dL/dy * sig(x) * (1 - sig(x))
    stats_.shaders_dispatched++;
    if (node->inputs[0].requires_grad) {
        node->grad_input_buffers[0] = 1;
    }
}

void BackwardEngine::backward_softmax(Node* node) {
    // Softmax backward: efficient Jacobian computation
    // dL/dx_i = s_i * (dL/dy_i - sum_j(dL/dy_j * s_j))
    stats_.shaders_dispatched++;
    if (node->inputs[0].requires_grad) {
        node->grad_input_buffers[0] = 1;
        // TODO: dispatch softmax backward shader
    }
}

void BackwardEngine::backward_layernorm(Node* node) {
    // LayerNorm backward: complex three-term gradient
    stats_.shaders_dispatched++;
    if (node->inputs[0].requires_grad) {
        node->grad_input_buffers[0] = 1;
        // TODO: dispatch layernorm-backward.glsl
    }
}

void BackwardEngine::backward_attention(Node* node) {
    // FlashAttention2 backward — most complex backward op
    // Requires saved Q, K, V, and the softmax statistics (m, l)
    stats_.shaders_dispatched++;
    for (uint32_t i = 0; i < node->num_inputs && i < 3; ++i) {
        if (node->inputs[i].requires_grad) {
            node->grad_input_buffers[i] = 1;
        }
    }
    // TODO: dispatch flash-attention2-backward.glsl (tiled)
}

void BackwardEngine::backward_conv2d(Node* node) {
    stats_.shaders_dispatched++;
    // Conv2d backward: input grad via transposed conv, weight grad via correlation
    if (node->inputs[0].requires_grad) {
        node->grad_input_buffers[0] = 1;
    }
    if (node->num_inputs > 1 && node->inputs[1].requires_grad) {
        node->grad_input_buffers[1] = 1;
    }
}

void BackwardEngine::backward_conv1d(Node* node) {
    stats_.shaders_dispatched++;
    if (node->inputs[0].requires_grad) {
        node->grad_input_buffers[0] = 1;
    }
    if (node->num_inputs > 1 && node->inputs[1].requires_grad) {
        node->grad_input_buffers[1] = 1;
    }
}

void BackwardEngine::backward_add(Node* node) {
    // Add: y = a + b → dL/da = dL/dy, dL/db = dL/dy (identity)
    // No shader needed — just pass through the gradient buffer
    for (uint32_t i = 0; i < node->num_inputs; ++i) {
        if (node->inputs[i].requires_grad) {
            node->grad_input_buffers[i] = node->grad_output_buffer;
        }
    }
}

void BackwardEngine::backward_sub(Node* node) {
    // Sub: y = a - b → dL/da = dL/dy, dL/db = -dL/dy
    if (node->inputs[0].requires_grad) {
        node->grad_input_buffers[0] = node->grad_output_buffer;
    }
    if (node->num_inputs > 1 && node->inputs[1].requires_grad) {
        node->grad_input_buffers[1] = 1;  // placeholder: negate shader
        stats_.shaders_dispatched++;
        // TODO: dispatch negation shader
    }
}

void BackwardEngine::backward_mul(Node* node) {
    // Mul: y = a * b → dL/da = dL/dy * b, dL/db = dL/dy * a
    stats_.shaders_dispatched++;
    if (node->inputs[0].requires_grad) {
        node->grad_input_buffers[0] = 1;
        // TODO: dispatch element-wise mul: grad_output * saved_b
    }
    if (node->num_inputs > 1 && node->inputs[1].requires_grad) {
        node->grad_input_buffers[1] = 1;
        // TODO: dispatch element-wise mul: grad_output * saved_a
    }
}

void BackwardEngine::backward_div(Node* node) {
    // Div: y = a / b
    // dL/da = dL/dy / b
    // dL/db = -dL/dy * a / b^2
    stats_.shaders_dispatched++;
    if (node->inputs[0].requires_grad) {
        node->grad_input_buffers[0] = 1;
    }
    if (node->num_inputs > 1 && node->inputs[1].requires_grad) {
        node->grad_input_buffers[1] = 1;
    }
}

void BackwardEngine::backward_cross_entropy(Node* node) {
    // Cross-entropy: combined softmax + NLL for numerical stability
    // dL/dx = softmax(x) - one_hot(target)
    stats_.shaders_dispatched++;
    if (node->inputs[0].requires_grad) {
        node->grad_input_buffers[0] = 1;
        // TODO: dispatch cross-entropy-backward.glsl
    }
}

void BackwardEngine::backward_mse(Node* node) {
    // MSE: L = mean((y_pred - y_true)^2)
    // dL/dy_pred = 2 * (y_pred - y_true) / N
    stats_.shaders_dispatched++;
    if (node->inputs[0].requires_grad) {
        node->grad_input_buffers[0] = 1;
    }
}

void BackwardEngine::backward_cubemind_surprise(Node* node) {
    // CubeMind Surprise-Momentum: modulate learning rate by VSA surprise.
    //
    // The surprise value from the hippocampal cache indicates how novel
    // the current input is. High surprise → larger gradient step (learn more),
    // low surprise → smaller step (already known).
    //
    // This is an EMA-based optimizer hook, not a traditional backward op.
    // It scales the incoming gradient by a surprise-derived multiplier:
    //   grad_modulated = grad_output * (1.0 + alpha * surprise)
    //
    // The emotion state (surprise, stress) was captured during the forward
    // pass from the VSA cache lookup and stored inline in the node.

    float surprise = node->emotion.surprise;
    float alpha = 0.5f;  // Surprise sensitivity — can be tuned

    // Read alpha from params if provided
    if (node->params_size >= sizeof(float)) {
        std::memcpy(&alpha, node->params, sizeof(float));
    }

    // The gradient modulation can be dispatched as a simple scalar-multiply
    // shader on the grad_output buffer, or done CPU-side for simplicity.
    if (node->inputs[0].requires_grad) {
        // Multiplier = 1 + alpha * surprise
        // High surprise → amplify gradient (learn more from novel inputs)
        // Low surprise → attenuate gradient (already known)
        float multiplier = 1.0f + alpha * surprise;

        // Store the multiplier in params for the scalar-multiply shader.
        // The actual dispatch uses: output[i] = input[i] * multiplier
        std::memcpy(node->params, &multiplier, sizeof(float));
        node->params_size = sizeof(float);

        node->grad_input_buffers[0] = node->grad_output_buffer;
        // TODO: dispatch scalar-multiply shader with push constant `multiplier`
        stats_.shaders_dispatched++;
    }
}

void BackwardEngine::backward_temporal_surprise(Node* node) {
    // Temporal Foresight: modulate gradient by counterfactual contradiction.
    //
    // During the forward pass, N counterfactual branches were evaluated:
    //   - Each branch: erase actual fact, insert "what if" fact, shift T+dt
    //   - Each branch checked against the WorldModel via Hamming search
    //
    // The TemporalSurpriseParams (stored in node->params) contain:
    //   avg_contradiction: mean surprise across all branches (0 = coherent, 1 = nonsense)
    //   temporal_multiplier: pre-computed 1.0 - 2.0 * avg_contradiction
    //
    // Gradient modulation:
    //   If futures are coherent (low contradiction) → multiplier ~1.0 (pass through)
    //   If futures are contradictory (high contradiction) → multiplier < 0 (penalize)
    //   The negative multiplier pushes weights AWAY from the incoherent trajectory.

    TemporalSurpriseParams tparams;
    std::memcpy(&tparams, node->params, sizeof(TemporalSurpriseParams));

    if (node->inputs[0].requires_grad) {
        float multiplier = tparams.temporal_multiplier * tparams.alpha;

        // Clamp to [-1, 1] to prevent gradient explosion
        if (multiplier > 1.0f) multiplier = 1.0f;
        if (multiplier < -1.0f) multiplier = -1.0f;

        // Store multiplier for scalar-multiply shader
        std::memcpy(node->params, &multiplier, sizeof(float));
        node->params_size = sizeof(float);

        node->grad_input_buffers[0] = node->grad_output_buffer;
        // TODO: dispatch scalar-multiply shader with push constant `multiplier`
        stats_.shaders_dispatched++;
    }
}

// Shape ops — no GPU shader needed, just logical reshaping of the gradient

void BackwardEngine::backward_reshape(Node* node) {
    // Reshape backward: gradient has the shape of the INPUT, not the output.
    // Since data layout in memory is unchanged, just pass the buffer through.
    if (node->inputs[0].requires_grad) {
        node->grad_input_buffers[0] = node->grad_output_buffer;
    }
}

void BackwardEngine::backward_transpose(Node* node) {
    // Transpose backward: transpose the gradient back.
    // For 2D: just swap dims. For ND: reverse the permutation.
    stats_.shaders_dispatched++;
    if (node->inputs[0].requires_grad) {
        node->grad_input_buffers[0] = 1;  // placeholder: transpose shader
        // TODO: dispatch transpose/permute shader
    }
}

void BackwardEngine::backward_sum(Node* node) {
    // Sum backward: gradient is broadcast-expanded to input shape
    if (node->inputs[0].requires_grad) {
        node->grad_input_buffers[0] = 1;  // placeholder: broadcast shader
        stats_.shaders_dispatched++;
    }
}

void BackwardEngine::backward_mean(Node* node) {
    // Mean backward: gradient = 1/N * ones (broadcast)
    if (node->inputs[0].requires_grad) {
        node->grad_input_buffers[0] = 1;
        stats_.shaders_dispatched++;
    }
}

// ═════════════════════════════════════════════════════════════════════════
// TapeContext
// ═════════════════════════════════════════════════════════════════════════

TapeContext::TapeContext(BufferPool& pool, CommandBatch& batch,
                         PipelineCache& cache, size_t arena_capacity)
    : arena_(arena_capacity),
      registry_(pool),
      engine_(pool, batch, cache, registry_) {}

void TapeContext::begin() {
    arena_.reset();
    registry_.clear();
    engine_.clear_grads();
    seq_counter_ = 0;
    recording_ = true;
}

Node* TapeContext::record_op(OpType op,
                              const TensorRef* inputs, uint32_t num_inputs,
                              const TensorRef* outputs, uint32_t num_outputs,
                              const void* params, uint32_t params_size) {
    if (!recording_) return nullptr;

    Node* node = arena_.allocate_node<Node>();

    node->op = op;
    node->seq = seq_counter_++;
    node->num_inputs = num_inputs;
    node->num_outputs = num_outputs;
    node->num_saved = 0;
    node->grad_output_buffer = 0;
    node->params_size = 0;
    node->emotion = {0.0f, 0.0f};

    // Zero out all IO slots
    std::memset(node->inputs, 0, sizeof(node->inputs));
    std::memset(node->outputs, 0, sizeof(node->outputs));
    std::memset(node->saved_buffer_ids, 0, sizeof(node->saved_buffer_ids));
    std::memset(node->grad_input_buffers, 0, sizeof(node->grad_input_buffers));

    // Copy input/output descriptors
    for (uint32_t i = 0; i < num_inputs && i < kMaxNodeIO; ++i) {
        node->inputs[i] = inputs[i];
    }
    for (uint32_t i = 0; i < num_outputs && i < kMaxNodeIO; ++i) {
        node->outputs[i] = outputs[i];
    }

    // Copy per-op parameters (push constant data)
    if (params && params_size > 0 && params_size <= sizeof(node->params)) {
        std::memcpy(node->params, params, params_size);
        node->params_size = params_size;
    }

    return node;
}

void TapeContext::save_for_backward(Node* node, const uint32_t* buffer_ids,
                                     uint32_t count) {
    if (!node) return;
    for (uint32_t i = 0; i < count && i < kMaxNodeIO; ++i) {
        node->saved_buffer_ids[i] = buffer_ids[i];
    }
    node->num_saved = std::min(count, static_cast<uint32_t>(kMaxNodeIO));
}

void TapeContext::backward(Node* loss_node, uint32_t grad_output_buffer) {
    recording_ = false;
    engine_.backward(arena_, loss_node, grad_output_buffer);
}

uint32_t TapeContext::get_grad_buffer(uint32_t input_buffer_id) const {
    return engine_.get_grad_buffer(input_buffer_id);
}

void TapeContext::end() {
    arena_.reset();
    engine_.clear_grads();
    seq_counter_ = 0;
    recording_ = false;
}

}  // namespace autograd
}  // namespace grilly
