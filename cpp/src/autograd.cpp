#include "grilly/autograd/autograd.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "grilly/ops/linear.h"
#include "grilly/ops/batched_ops.h"
#include "grilly/ops/loss.h"
#include "grilly/ops/rmsnorm.h"

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

        // Pull: a node activates if a downstream consumer has already
        // deposited a gradient under one of this node's OUTPUT buffer_ids.
        // (The walk is tail->head, so all consumers of `current` ran first.)
        // The loss node always activates: loss handlers (CE/MSE) compute their
        // own gradient from inputs+targets and don't read grad_output_buffer,
        // so a zero grad_output_buffer is fine for them.
        bool is_loss_node = (current == loss_node);
        if (current->grad_output_buffer == 0 && !is_loss_node) {
            for (uint32_t o = 0; o < current->num_outputs; ++o) {
                uint32_t out_buf = current->outputs[o].buffer_id;
                if (out_buf == 0) continue;
                uint32_t g = get_grad_buffer(out_buf);
                if (g != 0) {
                    current->grad_output_buffer = g;
                    break;
                }
            }
        }

        if (current->grad_output_buffer != 0 || is_loss_node) {
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
                    // Fan-out: accum += new_grad (in place), via the existing
                    // elementwise-add shader. Both ids resolve through the
                    // registry to resident buffers; no CPU round-trip.
                    uint32_t numel =
                        static_cast<uint32_t>(current->inputs[i].numel());
                    GrillyBuffer& accum_buf = registry_.resolve(accum);
                    GrillyBuffer& new_grad_buf =
                        registry_.resolve(current->grad_input_buffers[i]);
                    grilly::ops::batchedAdd(batch_, cache_, accum_buf,
                                            new_grad_buf, numel);
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
        case OpType::RMSNorm:   backward_rmsnorm(node); break;
        case OpType::MinGRU:    backward_mingru(node); break;
        case OpType::SwiGLU:    backward_swiglu(node); break;
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

void BackwardEngine::backward_activation(Node* node, const char* shaderName) {
    // Pointwise activation backward. Shader takes 3 buffers:
    //   binding 0: grad_output (dL/dy, resident incoming gradient)
    //   binding 1: input       (saved pre-activation x)
    //   binding 2: grad_input  (dL/dx, output)
    // push = ActivationParams{totalElements}; gx = (N+255)/256.
    if (!node->inputs[0].requires_grad) return;
    if (node->grad_output_buffer == 0) { stats_.cpu_fallbacks++; return; }

    uint32_t inId = (node->num_saved > 0 && node->saved_buffer_ids[0] != 0)
                        ? node->saved_buffer_ids[0] : node->inputs[0].buffer_id;
    if (inId == 0) { stats_.cpu_fallbacks++; return; }

    uint32_t totalElements = static_cast<uint32_t>(node->inputs[0].numel());
    const size_t bytes = size_t(totalElements) * 4;

    GrillyBuffer& gradOut = registry_.resolve(node->grad_output_buffer);
    GrillyBuffer& inBuf   = registry_.resolve(inId);
    uint32_t gradInId = registry_.alloc(bytes);
    GrillyBuffer& gradInBuf = registry_.resolve(gradInId);

    PipelineEntry pipe = cache_.getOrCreate(shaderName, 3, sizeof(uint32_t));
    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {gradOut.handle,   0, bytes},
        {inBuf.handle,     0, bytes},
        {gradInBuf.handle, 0, bytes},
    };
    VkDescriptorSet descSet = cache_.allocDescriptorSet(shaderName, bufInfos);

    uint32_t push = totalElements;
    uint32_t gx = (totalElements + 255) / 256;
    batch_.dispatch(pipe.pipeline, pipe.layout, descSet, gx, 1, 1,
                    &push, sizeof(push));

    stats_.shaders_dispatched++;
    node->grad_input_buffers[0] = gradInId;
}

void BackwardEngine::backward_relu(Node* node) {
    // ReLU: dL/dx = dL/dy * (x > 0). Uses saved pre-activation x.
    backward_activation(node, "activation-relu-backward");
}

void BackwardEngine::backward_gelu(Node* node) {
    // GELU backward — uses the pre-activation value.
    backward_activation(node, "activation-gelu-backward");
}

void BackwardEngine::backward_silu(Node* node) {
    // SiLU: dL/dx = dL/dy * (sigmoid(x) + x*sigmoid(x)*(1-sigmoid(x))).
    backward_activation(node, "activation-silu-backward");
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

void BackwardEngine::backward_rmsnorm(Node* node) {
    // RMSNorm: y_i = w_f * x_i * r,  r = inversesqrt(mean(x^2) + eps).
    // 2-pass rms-norm-backward shader (no atomics, one thread per output cell):
    //   pass 0: grad_input  (batch*seq*features threads)
    //   pass 1: grad_weight (features threads)
    //
    // inputs[0] = x (B*S, features), inputs[1] = weight (features,)
    // saved_buffer_ids[0] = x, saved_buffer_ids[1] = weight
    if (!node->inputs[0].requires_grad &&
        !(node->num_inputs > 1 && node->inputs[1].requires_grad)) {
        return;
    }

    const TensorRef& xRef = node->inputs[0];
    if (xRef.ndim < 1) { stats_.cpu_fallbacks++; return; }
    const uint32_t features = xRef.shape[xRef.ndim - 1];
    if (features == 0) { stats_.cpu_fallbacks++; return; }
    const uint32_t totalPositions =
        static_cast<uint32_t>(xRef.numel() / features);
    const uint32_t totalElements = totalPositions * features;

    uint32_t xId = (node->num_saved > 0 && node->saved_buffer_ids[0] != 0)
                       ? node->saved_buffer_ids[0] : xRef.buffer_id;
    uint32_t wId = (node->num_saved > 1 && node->saved_buffer_ids[1] != 0)
                       ? node->saved_buffer_ids[1]
                       : (node->num_inputs > 1 ? node->inputs[1].buffer_id : 0);
    if (xId == 0 || wId == 0 || node->grad_output_buffer == 0) {
        stats_.cpu_fallbacks++; return;
    }

    GrillyBuffer& gradOut = registry_.resolve(node->grad_output_buffer);
    GrillyBuffer& xBuf = registry_.resolve(xId);
    GrillyBuffer& wBuf = registry_.resolve(wId);

    const size_t elemBytes = size_t(totalElements) * 4;
    const size_t featBytes = size_t(features) * 4;

    uint32_t gradXId = registry_.alloc(elemBytes);
    uint32_t gradWId = registry_.alloc(featBytes);
    GrillyBuffer& gradXBuf = registry_.resolve(gradXId);
    GrillyBuffer& gradWBuf = registry_.resolve(gradWId);

    PipelineEntry pipe = cache_.getOrCreate(
        "rms-norm-backward", 5, sizeof(grilly::ops::RMSNormParams));
    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {gradOut.handle,  0, elemBytes},
        {xBuf.handle,     0, elemBytes},
        {wBuf.handle,     0, featBytes},
        {gradXBuf.handle, 0, elemBytes},
        {gradWBuf.handle, 0, featBytes},
    };
    VkDescriptorSet descSet =
        cache_.allocDescriptorSet("rms-norm-backward", bufInfos);

    // The op encodes (batch, seq); the shader only uses batch_size*seq_len, so
    // pack totalPositions into batch_size and seq_len=1. eps default 1e-6.
    grilly::ops::RMSNormParams p{};
    p.batchSize = totalPositions;
    p.seqLen = 1;
    p.features = features;
    p.eps = 1e-6f;

    // Pass 0: grad_input
    p.passType = 0;
    batch_.dispatch(pipe.pipeline, pipe.layout, descSet,
                    (totalElements + 255) / 256, 1, 1, &p, sizeof(p));
    batch_.barrier();

    // Pass 1: grad_weight
    p.passType = 1;
    batch_.dispatch(pipe.pipeline, pipe.layout, descSet,
                    (features + 255) / 256, 1, 1, &p, sizeof(p));

    stats_.shaders_dispatched++;
    if (xRef.requires_grad) node->grad_input_buffers[0] = gradXId;
    if (node->num_inputs > 1 && node->inputs[1].requires_grad)
        node->grad_input_buffers[1] = gradWId;
}

void BackwardEngine::backward_mingru(Node* node) {
    // Fused MinGRU backward. Forward:
    //   x_scan = sigmoid(g)*tanh(v);  a = 0.05+0.9*sigmoid(d)
    //   h_t = a_t*h_{t-1} + x_scan_t
    // Shader mingru-backward: 8 buffers {gradH, G, V, D, H, gradG, gradV, gradD},
    // push {seqLen, hiddenDim}, dispatch gx=(hiddenDim+63)/64, gy=batchSize.
    //
    // inputs[0]=G, inputs[1]=V, inputs[2]=D (the three projections).
    // saved_buffer_ids = [G, V, D, H]  (H = forward output, needed for backward).
    if (node->num_inputs < 3) { stats_.cpu_fallbacks++; return; }
    if (node->grad_output_buffer == 0) { stats_.cpu_fallbacks++; return; }
    if (node->num_saved < 4) { stats_.cpu_fallbacks++; return; }

    const TensorRef& gRef = node->inputs[0];
    // G is (batch, seq, hidden). Derive dims from shape: last = hidden.
    if (gRef.ndim < 1) { stats_.cpu_fallbacks++; return; }
    const uint32_t hiddenDim = gRef.shape[gRef.ndim - 1];
    if (hiddenDim == 0) { stats_.cpu_fallbacks++; return; }
    uint32_t seqLen = 1, batchSize = 1;
    if (gRef.ndim >= 3) {
        seqLen = gRef.shape[gRef.ndim - 2];
        batchSize = static_cast<uint32_t>(gRef.numel() / (size_t(seqLen) * hiddenDim));
    } else {
        // Fall back to (positions, hidden) with seq packed in batch.
        batchSize = static_cast<uint32_t>(gRef.numel() / hiddenDim);
        seqLen = 1;
    }

    GrillyBuffer& gradH = registry_.resolve(node->grad_output_buffer);
    GrillyBuffer& gBuf = registry_.resolve(node->saved_buffer_ids[0]);
    GrillyBuffer& vBuf = registry_.resolve(node->saved_buffer_ids[1]);
    GrillyBuffer& dBuf = registry_.resolve(node->saved_buffer_ids[2]);
    GrillyBuffer& hBuf = registry_.resolve(node->saved_buffer_ids[3]);

    const size_t elemBytes = size_t(batchSize) * seqLen * hiddenDim * 4;
    uint32_t gradGId = registry_.alloc(elemBytes);
    uint32_t gradVId = registry_.alloc(elemBytes);
    uint32_t gradDId = registry_.alloc(elemBytes);
    GrillyBuffer& gradGBuf = registry_.resolve(gradGId);
    GrillyBuffer& gradVBuf = registry_.resolve(gradVId);
    GrillyBuffer& gradDBuf = registry_.resolve(gradDId);

    PipelineEntry pipe =
        cache_.getOrCreate("mingru-backward", 8, 2 * sizeof(uint32_t));
    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {gradH.handle,    0, elemBytes},
        {gBuf.handle,     0, elemBytes},
        {vBuf.handle,     0, elemBytes},
        {dBuf.handle,     0, elemBytes},
        {hBuf.handle,     0, elemBytes},
        {gradGBuf.handle, 0, elemBytes},
        {gradVBuf.handle, 0, elemBytes},
        {gradDBuf.handle, 0, elemBytes},
    };
    VkDescriptorSet descSet = cache_.allocDescriptorSet("mingru-backward", bufInfos);

    struct { uint32_t seqLen; uint32_t hiddenDim; } push = {seqLen, hiddenDim};
    uint32_t gx = (hiddenDim + 63u) / 64u;
    uint32_t gy = batchSize;
    batch_.dispatch(pipe.pipeline, pipe.layout, descSet, gx, gy, 1,
                    &push, sizeof(push));

    stats_.shaders_dispatched++;
    if (gRef.requires_grad) node->grad_input_buffers[0] = gradGId;
    if (node->inputs[1].requires_grad) node->grad_input_buffers[1] = gradVId;
    if (node->inputs[2].requires_grad) node->grad_input_buffers[2] = gradDId;
}

void BackwardEngine::backward_swiglu(Node* node) {
    // SwiGLU: out = x1 * silu(x2). Input is [x1:hidden][x2:hidden] concatenated
    // (last dim = 2*hidden); output is hidden wide.
    // Shader activation-swiglu-backward: 3 buffers
    //   {grad_output (output_elements), input (2*hidden), grad_input (2*hidden)},
    //   push {output_elements, hidden_dim}, gx=(output_elements+255)/256.
    //
    // inputs[0] = concatenated [x1|x2]; saved_buffer_ids[0] = that input.
    if (!node->inputs[0].requires_grad) return;
    if (node->grad_output_buffer == 0) { stats_.cpu_fallbacks++; return; }

    const TensorRef& inRef = node->inputs[0];
    if (inRef.ndim < 1) { stats_.cpu_fallbacks++; return; }
    const uint32_t inWidth = inRef.shape[inRef.ndim - 1];   // = 2*hidden
    if (inWidth == 0 || (inWidth & 1u)) { stats_.cpu_fallbacks++; return; }
    const uint32_t hiddenDim = inWidth / 2u;
    const uint32_t rows =
        static_cast<uint32_t>(inRef.numel() / inWidth);
    const uint32_t outputElements = rows * hiddenDim;

    uint32_t inId = (node->num_saved > 0 && node->saved_buffer_ids[0] != 0)
                        ? node->saved_buffer_ids[0] : inRef.buffer_id;
    if (inId == 0) { stats_.cpu_fallbacks++; return; }

    GrillyBuffer& gradOut = registry_.resolve(node->grad_output_buffer);
    GrillyBuffer& inBuf = registry_.resolve(inId);
    const size_t inBytes = size_t(rows) * inWidth * 4;
    const size_t outBytes = size_t(outputElements) * 4;
    uint32_t gradInId = registry_.alloc(inBytes);
    GrillyBuffer& gradInBuf = registry_.resolve(gradInId);

    PipelineEntry pipe =
        cache_.getOrCreate("activation-swiglu-backward", 3, 2 * sizeof(uint32_t));
    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {gradOut.handle,   0, outBytes},
        {inBuf.handle,     0, inBytes},
        {gradInBuf.handle, 0, inBytes},
    };
    VkDescriptorSet descSet =
        cache_.allocDescriptorSet("activation-swiglu-backward", bufInfos);

    struct { uint32_t outputElements; uint32_t hiddenDim; } push =
        {outputElements, hiddenDim};
    uint32_t gx = (outputElements + 255u) / 256u;
    batch_.dispatch(pipe.pipeline, pipe.layout, descSet, gx, 1, 1,
                    &push, sizeof(push));

    stats_.shaders_dispatched++;
    node->grad_input_buffers[0] = gradInId;
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
    // Cross-entropy (combined softmax + NLL). This is a loss node: it has no
    // incoming grad_output buffer; the gradient w.r.t. logits is computed
    // directly from logits and targets:
    //   dL/dx = softmax(x) - one_hot(target)
    //
    // inputs[0] = logits (batchSize, numClasses), requires_grad
    // saved_buffer_ids[0] = logits buffer id
    // saved_buffer_ids[1] = targets buffer id (uint32 class indices)
    if (!node->inputs[0].requires_grad) return;

    const TensorRef& logitsRef = node->inputs[0];
    if (logitsRef.ndim < 1) { stats_.cpu_fallbacks++; return; }

    const uint32_t numClasses = logitsRef.shape[logitsRef.ndim - 1];
    if (numClasses == 0) { stats_.cpu_fallbacks++; return; }
    const uint32_t batchSize =
        static_cast<uint32_t>(logitsRef.numel() / numClasses);

    uint32_t logitsId = (node->num_saved > 0 && node->saved_buffer_ids[0] != 0)
                            ? node->saved_buffer_ids[0] : logitsRef.buffer_id;
    uint32_t targetsId = (node->num_saved > 1) ? node->saved_buffer_ids[1] : 0;
    if (logitsId == 0 || targetsId == 0) { stats_.cpu_fallbacks++; return; }

    GrillyBuffer& logitsBuf = registry_.resolve(logitsId);
    GrillyBuffer& targetsBuf = registry_.resolve(targetsId);

    const size_t logitBytes = size_t(batchSize) * numClasses * 4;
    const size_t targetBytes = size_t(batchSize) * 4;

    uint32_t gradId = registry_.alloc(logitBytes);
    GrillyBuffer& gradBuf = registry_.resolve(gradId);

    PipelineEntry pipe = cache_.getOrCreate(
        "cross-entropy-backward", 3, sizeof(grilly::ops::CrossEntropyBackwardParams));
    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {logitsBuf.handle,  0, logitBytes},
        {targetsBuf.handle, 0, targetBytes},
        {gradBuf.handle,    0, logitBytes},
    };
    VkDescriptorSet descSet =
        cache_.allocDescriptorSet("cross-entropy-backward", bufInfos);

    grilly::ops::CrossEntropyBackwardParams p{batchSize, numClasses};
    // The shader uses one workgroup per batch row (batch_idx = gl_WorkGroupID.x),
    // so dispatch exactly batchSize workgroups.
    uint32_t gx = batchSize;
    batch_.dispatch(pipe.pipeline, pipe.layout, descSet, gx, 1, 1,
                    &p, sizeof(p));

    stats_.shaders_dispatched++;
    node->grad_input_buffers[0] = gradId;
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
