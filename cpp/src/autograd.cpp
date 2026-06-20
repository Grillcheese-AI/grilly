#include "grilly/autograd/autograd.h"

#include <algorithm>
#include <chrono>
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

            // Barrier BEFORE dispatch: this node reads grad_output_buffer, which
            // may have been built by accumulation (batchedAdd) writes while
            // processing downstream nodes. Those writes must be visible before
            // this node's shader reads them. (Without this, e.g. RMSNorm reading
            // an n1 grad accumulated from three Linear branches races the adds.)
            batch_.transferComputeBarrier();

            // Dispatch the backward shader for this operation
            dispatch_node_backward(current);

            // Barrier: the handler's grad writes must be visible before any
            // accumulation reads them (and before downstream nodes read them).
            // Use transferComputeBarrier (not plain barrier) because some
            // handlers (e.g. backward_linear) zero grad buffers with fillZero,
            // a TRANSFER_WRITE; a compute-only barrier wouldn't order that
            // against a downstream node's shader read (the MinGRU<-Linear bug).
            batch_.transferComputeBarrier();

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
                    // NOTE: when the SAME input buffer feeds multiple inputs of
                    // one node (e.g. MinGRU with G==V==D), this loop runs several
                    // times read-modify-writing `accum`; a barrier before each
                    // add orders them so the contributions sum correctly.
                    batch_.transferComputeBarrier();
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
        case OpType::ChunkedAttention: backward_chunked_attention(node); break;
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

    // grad_weight/grad_bias write directly but zero for safety.
    batch_.fillZero(gradWBuf, gradWBytes);
    batch_.fillZero(gradBiasBuf, gradBiasBytes);
    batch_.transferComputeBarrier();

    grilly::ops::LinearBackwardParams p{batchSeq, inputDim, outputDim, 0};

    // Pass 0: grad_input = grad_output @ W -- via the TILED GEMM (gemm_tiled),
    // NOT the naive fnn-linear-backward pass (which had each thread serially loop
    // output_dim at low occupancy -- the v3.3 backward bottleneck, esp. the
    // V=65536 head). grad_out (M=batchSeq, K=outputDim) @ W (K=outputDim,
    // N=inputDim, stored out x in) = grad_input (M x N). fp32, exact.
    {
        PipelineEntry gpipe =
            cache_.getOrCreate("gemm_tiled", 3, 3 * sizeof(uint32_t));
        std::vector<VkDescriptorBufferInfo> gbuf = {
            {gradOut.handle,   0, gradOutBytes},   // A = grad_output (M x K)
            {wBuf.handle,      0, gradWBytes},     // B = W (K x N)
            {gradInBuf.handle, 0, gradInBytes},    // C = grad_input (M x N)
        };
        VkDescriptorSet gds = cache_.allocDescriptorSet("gemm_tiled", gbuf);
        struct { uint32_t M, K, N; } gp = {batchSeq, outputDim, inputDim};
        batch_.dispatch(gpipe.pipeline, gpipe.layout, gds,
                        (inputDim + 63u) / 64u, (batchSeq + 63u) / 64u, 1,
                        &gp, sizeof(gp));
    }
    batch_.barrier();

    // Pass 1: grad_weight = grad_out^T @ x -- TILED, was the naive memory-bound
    // pass (strided reads over batchSeq per thread). transpose grad_out
    // (batchSeq x outputDim) -> (outputDim x batchSeq), then gemm_tiled
    // (M=outputDim, K=batchSeq, N=inputDim).
    uint32_t gradOutTId = registry_.alloc(gradOutBytes);
    GrillyBuffer& gradOutTBuf = registry_.resolve(gradOutTId);
    {
        PipelineEntry tpipe =
            cache_.getOrCreate("tensor-transpose", 2, 2 * sizeof(uint32_t));
        std::vector<VkDescriptorBufferInfo> tbuf = {
            {gradOut.handle,     0, gradOutBytes},
            {gradOutTBuf.handle, 0, gradOutBytes},
        };
        VkDescriptorSet tds = cache_.allocDescriptorSet("tensor-transpose", tbuf);
        struct { uint32_t rows, cols; } tp = {batchSeq, outputDim};
        batch_.dispatch(tpipe.pipeline, tpipe.layout, tds,
                        (batchSeq * outputDim + 255u) / 256u, 1, 1, &tp, sizeof(tp));
    }
    batch_.barrier();
    {
        PipelineEntry gpipe =
            cache_.getOrCreate("gemm_tiled", 3, 3 * sizeof(uint32_t));
        std::vector<VkDescriptorBufferInfo> gbuf = {
            {gradOutTBuf.handle, 0, gradOutBytes},                     // A = grad_out^T (out x BS)
            {xBuf.handle,        0, size_t(batchSeq) * inputDim * 4},  // B = x (BS x in)
            {gradWBuf.handle,    0, gradWBytes},                       // C = grad_weight (out x in)
        };
        VkDescriptorSet gds = cache_.allocDescriptorSet("gemm_tiled", gbuf);
        struct { uint32_t M, K, N; } gp = {outputDim, batchSeq, inputDim};
        batch_.dispatch(gpipe.pipeline, gpipe.layout, gds,
                        (inputDim + 63u) / 64u, (outputDim + 63u) / 64u, 1,
                        &gp, sizeof(gp));
    }
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

void BackwardEngine::backward_chunked_attention(Node* node) {
    // Sliding-window causal attention backward (0.0.2) — GPU-side.
    //
    // inputs[0] = qkv (BS, 3*d) — the fused QKV projection output.
    // saved_buffer_ids = [q_id, k_id, v_id] — the split Q/K/V buffers (BHSD).
    // grad_output_buffer = dO (B, H, S, Dh) — gradient w.r.t. attention output.
    //
    // GPU backward: dispatch chunked-sw-attention-backward.glsl (recomputes
    // softmax on-the-fly, uses atomicAdd for dK/dV), then merge dQ/dK/dV
    // back into d_qkv via attention-qkv-merge.glsl.

    if (node->num_inputs < 1) return;
    if (node->grad_output_buffer == 0) { stats_.cpu_fallbacks++; return; }
    if (node->num_saved < 3) { stats_.cpu_fallbacks++; return; }
    if (!node->inputs[0].requires_grad) return;

    // Unpack params
    struct { uint32_t B; uint32_t H; uint32_t S; uint32_t Dh; uint32_t W; float scale; } p;
    std::memcpy(&p, node->params, std::min<size_t>(node->params_size, sizeof(p)));
    const uint32_t B = p.B, H = p.H, S = p.S, Dh = p.Dh, W = p.W;
    const float scale = p.scale;
    const size_t elemBytes = size_t(B) * H * S * Dh * 4;
    const uint32_t d = H * Dh;
    const size_t qkvBytes = size_t(B) * S * 3 * d * 4;

    uint32_t qId = node->saved_buffer_ids[0];
    uint32_t kId = node->saved_buffer_ids[1];
    uint32_t vId = node->saved_buffer_ids[2];
    uint32_t dOId = node->grad_output_buffer;

    // Allocate dQ, dK, dV (BHSD layout)
    uint32_t dQId = registry_.alloc(elemBytes);
    uint32_t dKId = registry_.alloc(elemBytes);
    uint32_t dVId = registry_.alloc(elemBytes);

    // Zero dK and dV (they use atomicAdd) — via batch_.fillZero, not registry_.upload
    GrillyBuffer& dKBufZ = registry_.resolve(dKId);
    GrillyBuffer& dVBufZ = registry_.resolve(dVId);
    batch_.fillZero(dKBufZ, elemBytes);
    batch_.fillZero(dVBufZ, elemBytes);
    batch_.transferComputeBarrier();

    // ── Dispatch backward shader ───────────────────────────────────────
    GrillyBuffer& qBuf = registry_.resolve(qId);
    GrillyBuffer& kBuf = registry_.resolve(kId);
    GrillyBuffer& vBuf = registry_.resolve(vId);
    GrillyBuffer& dOBuf = registry_.resolve(dOId);
    GrillyBuffer& dQBuf = registry_.resolve(dQId);
    GrillyBuffer& dKBuf = registry_.resolve(dKId);
    GrillyBuffer& dVBuf = registry_.resolve(dVId);

    constexpr uint32_t kNumBindingsBwd = 7;
    constexpr uint32_t kPushConstBytes = 24;  // 6 * sizeof(uint32_t) — but scale is float
    PipelineEntry pipeBwd = cache_.getOrCreate("chunked-sw-attention-backward",
                                               kNumBindingsBwd, kPushConstBytes);
    std::vector<VkDescriptorBufferInfo> bufInfosBwd = {
        {qBuf.handle,  0, elemBytes},
        {kBuf.handle,  0, elemBytes},
        {vBuf.handle,  0, elemBytes},
        {dOBuf.handle, 0, elemBytes},
        {dQBuf.handle, 0, elemBytes},
        {dKBuf.handle, 0, elemBytes},
        {dVBuf.handle, 0, elemBytes},
    };
    VkDescriptorSet descSetBwd = cache_.allocDescriptorSet(
        "chunked-sw-attention-backward", bufInfosBwd);

    struct PushBwd {
        uint32_t batch_size, num_heads, seq_len, head_dim, window_size;
        float scale;
    } pushBwd = {B, H, S, Dh, W, scale};

    const uint32_t gxBwd = B * H * S;
    batch_.dispatch(pipeBwd.pipeline, pipeBwd.layout, descSetBwd,
                    gxBwd, 1, 1, &pushBwd, sizeof(pushBwd));
    batch_.transferComputeBarrier();

    // ── Merge dQ/dK/dV into d_qkv (BS, 3*d) BSHD layout ───────────────
    uint32_t dQkvId = registry_.alloc(qkvBytes);
    GrillyBuffer& dQkvBuf = registry_.resolve(dQkvId);

    constexpr uint32_t kNumBindingsMerge = 4;
    constexpr uint32_t kPushConstBytesMerge = 16;
    PipelineEntry pipeMerge = cache_.getOrCreate("attention-qkv-merge",
                                                  kNumBindingsMerge,
                                                  kPushConstBytesMerge);
    std::vector<VkDescriptorBufferInfo> bufInfosMerge = {
        {dQBuf.handle,  0, elemBytes},
        {dKBuf.handle,  0, elemBytes},
        {dVBuf.handle,  0, elemBytes},
        {dQkvBuf.handle, 0, qkvBytes},
    };
    VkDescriptorSet descSetMerge = cache_.allocDescriptorSet(
        "attention-qkv-merge", bufInfosMerge);

    struct PushMerge {
        uint32_t batch_size, num_heads, seq_len, head_dim;
    } pushMerge = {B, H, S, Dh};

    const uint32_t totalMerge = B * H * S * Dh;
    batch_.dispatch(pipeMerge.pipeline, pipeMerge.layout, descSetMerge,
                    (totalMerge + 63) / 64, 1, 1, &pushMerge, sizeof(pushMerge));
    batch_.transferComputeBarrier();

    stats_.shaders_dispatched += 2;
    node->grad_input_buffers[0] = dQkvId;
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
    // BHSD <-> BSHD transpose backward. GPU dispatch: the transpose is its own
    // inverse (swap dims 1 and 2). We reuse the attention-transpose-bhsd-bshd
    // shader which does exactly this permutation.
    if (node->grad_output_buffer == 0) { stats_.cpu_fallbacks++; return; }

    const TensorRef& inRef = node->inputs[0];
    if (inRef.ndim < 4) {
        // Fallback: pass through for simple reshapes
        if (inRef.requires_grad) node->grad_input_buffers[0] = node->grad_output_buffer;
        return;
    }

    const uint32_t B = inRef.shape[0];
    const uint32_t H = inRef.shape[1];
    const uint32_t S = inRef.shape[2];
    const uint32_t Dh = inRef.shape[3];
    const size_t elemBytes = size_t(B) * H * S * Dh * 4;

    GrillyBuffer& gOBuf = registry_.resolve(node->grad_output_buffer);
    uint32_t gradInId = registry_.alloc(elemBytes);
    GrillyBuffer& gradInBuf = registry_.resolve(gradInId);

    PipelineEntry pipe = cache_.getOrCreate("attention-transpose-bhsd-bshd",
                                            2, 16);
    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {gOBuf.handle,    0, elemBytes},
        {gradInBuf.handle, 0, elemBytes},
    };
    VkDescriptorSet descSet = cache_.allocDescriptorSet(
        "attention-transpose-bhsd-bshd", bufInfos);

    struct { uint32_t batch, heads, seq, dh; } push = {B, H, S, Dh};
    const uint32_t total = B * H * S * Dh;
    batch_.dispatch(pipe.pipeline, pipe.layout, descSet,
                    (total + 63) / 64, 1, 1, &push, sizeof(push));
    batch_.transferComputeBarrier();

    stats_.shaders_dispatched++;
    if (inRef.requires_grad) node->grad_input_buffers[0] = gradInId;
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
      engine_(pool, batch, cache, registry_),
      batch_(batch),
      cache_(cache) {}

void TapeContext::begin() {
    arena_.reset();
    registry_.clear();
    // Free cached descriptor sets between steps: the buffer pool recycles handles
    // each step (step-scoped buffers are released by registry_.clear() above), so
    // a descriptor set cached under an old handle would be falsely reused for a
    // different buffer. Safe here -- every batch submit() is synchronous, so the
    // prior step's GPU work has completed.
    cache_.clearDescriptorCache();
    engine_.clear_grads();
    seq_counter_ = 0;
    recording_ = true;
}

// ── Resident forward pass ────────────────────────────────────────────────

void TapeContext::forward_begin() {
    batch_.begin();
}

void TapeContext::forward_submit() {
    batch_.submit();
}

uint32_t TapeContext::forward_linear(uint32_t in_id, uint32_t weight_id,
                                     uint32_t bias_id, uint32_t M, uint32_t K,
                                     uint32_t N) {
    // out = in @ weight^T (+ bias). in (M,K), weight (N,K) -> out (M,N).
    const size_t outBytes = size_t(M) * N * 4;
    uint32_t outId = registry_.alloc(outBytes);

    GrillyBuffer& inBuf = registry_.resolve(in_id);
    GrillyBuffer& wBuf = registry_.resolve(weight_id);
    GrillyBuffer& outBuf = registry_.resolve(outId);
    const GrillyBuffer* biasPtr =
        (bias_id != 0) ? &registry_.resolve(bias_id) : nullptr;

    grilly::ops::batchedLinear(batch_, cache_, inBuf, wBuf, biasPtr, outBuf,
                               M, K, N);
    // Order this dispatch before any consumer reads outId in the same batch.
    batch_.transferComputeBarrier();
    return outId;
}

uint32_t TapeContext::forward_rmsnorm(uint32_t in_id, uint32_t weight_id,
                                      uint32_t positions, uint32_t features) {
    // out = weight * x * rsqrt(mean(x^2)+eps). 2-pass rms-norm shader:
    //   4 buffers {input, output, weight, rms_vals(positions)}.
    //   pass 0 (gx=positions): rms_vals[p] = mean(x^2)
    //   pass 1 (gx=positions*features): normalize.
    const uint32_t totalElements = positions * features;
    const size_t elemBytes = size_t(totalElements) * 4;
    const size_t rmsBytes = size_t(positions) * 4;

    uint32_t outId = registry_.alloc(elemBytes);
    uint32_t rmsId = registry_.alloc(rmsBytes);
    GrillyBuffer& inBuf = registry_.resolve(in_id);
    GrillyBuffer& outBuf = registry_.resolve(outId);
    GrillyBuffer& wBuf = registry_.resolve(weight_id);
    GrillyBuffer& rmsBuf = registry_.resolve(rmsId);

    PipelineEntry pipe = cache_.getOrCreate(
        "rms-norm", 4, sizeof(grilly::ops::RMSNormParams));
    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {inBuf.handle,  0, elemBytes},
        {outBuf.handle, 0, elemBytes},
        {wBuf.handle,   0, size_t(features) * 4},
        {rmsBuf.handle, 0, rmsBytes},
    };
    VkDescriptorSet descSet = cache_.allocDescriptorSet("rms-norm", bufInfos);

    grilly::ops::RMSNormParams p{};
    p.batchSize = positions;
    p.seqLen = 1;
    p.features = features;
    p.eps = 1e-6f;

    p.passType = 0;
    batch_.dispatch(pipe.pipeline, pipe.layout, descSet,
                    (positions + 255) / 256, 1, 1, &p, sizeof(p));
    batch_.barrier();
    p.passType = 1;
    batch_.dispatch(pipe.pipeline, pipe.layout, descSet,
                    (totalElements + 255) / 256, 1, 1, &p, sizeof(p));
    batch_.transferComputeBarrier();
    return outId;
}

uint32_t TapeContext::forward_swiglu(uint32_t in_id, uint32_t rows,
                                     uint32_t hidden) {
    // out = x1*silu(x2). input [x1|x2] (rows, 2*hidden) -> out (rows, hidden).
    // 2 buffers {input(2*hidden), output(hidden)}, push {output_elements, hidden}.
    const uint32_t outputElements = rows * hidden;
    const size_t inBytes = size_t(rows) * (2u * hidden) * 4;
    const size_t outBytes = size_t(outputElements) * 4;

    uint32_t outId = registry_.alloc(outBytes);
    GrillyBuffer& inBuf = registry_.resolve(in_id);
    GrillyBuffer& outBuf = registry_.resolve(outId);

    PipelineEntry pipe =
        cache_.getOrCreate("activation-swiglu", 2, 2 * sizeof(uint32_t));
    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {inBuf.handle,  0, inBytes},
        {outBuf.handle, 0, outBytes},
    };
    VkDescriptorSet descSet = cache_.allocDescriptorSet("activation-swiglu", bufInfos);

    struct { uint32_t outputElements; uint32_t hidden; } push = {outputElements, hidden};
    batch_.dispatch(pipe.pipeline, pipe.layout, descSet,
                    (outputElements + 255) / 256, 1, 1, &push, sizeof(push));
    batch_.transferComputeBarrier();
    return outId;
}

uint32_t TapeContext::forward_mingru(uint32_t g_id, uint32_t v_id, uint32_t d_id,
                                     uint32_t batch, uint32_t seqLen,
                                     uint32_t hidden) {
    // H = MinGRU(G,V,D). G/V/D/H all (batch, seqLen, hidden), laid out [b][t][d].
    // Fused activation + causal scan via mingru-forward.glsl: one thread per
    // (batch, hidden), sequential time loop. x_scan=sigmoid(g)*tanh(v),
    // a=0.001+0.998*sigmoid(d), h_t=a*h_{t-1}+x_scan -- matches backward_mingru
    // and the numpy reference exactly.
    const size_t elemBytes = size_t(batch) * seqLen * hidden * 4;
    uint32_t outId = registry_.alloc(elemBytes);
    GrillyBuffer& gBuf = registry_.resolve(g_id);
    GrillyBuffer& vBuf = registry_.resolve(v_id);
    GrillyBuffer& dBuf = registry_.resolve(d_id);
    GrillyBuffer& outBuf = registry_.resolve(outId);

    PipelineEntry pipe =
        cache_.getOrCreate("mingru-forward", 4, 2 * sizeof(uint32_t));
    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {gBuf.handle,   0, elemBytes},
        {vBuf.handle,   0, elemBytes},
        {dBuf.handle,   0, elemBytes},
        {outBuf.handle, 0, elemBytes},
    };
    VkDescriptorSet descSet =
        cache_.allocDescriptorSet("mingru-forward", bufInfos);

    struct { uint32_t seqLen; uint32_t hiddenDim; } push = {seqLen, hidden};
    uint32_t gx = (hidden + 63u) / 64u;
    uint32_t gy = batch;
    batch_.dispatch(pipe.pipeline, pipe.layout, descSet, gx, gy, 1,
                    &push, sizeof(push));
    batch_.transferComputeBarrier();
    return outId;
}

uint32_t TapeContext::forward_chunked_attention(
    uint32_t q_id, uint32_t k_id, uint32_t v_id,
    uint32_t batch, uint32_t num_heads, uint32_t seq_len, uint32_t head_dim,
    uint32_t window_size) {
    // Sliding-window causal attention (0.0.2).
    // Q/K/V/O all (batch, num_heads, seq_len, head_dim), BHSD layout.
    // chunked-sw-attention.glsl: one workgroup per (batch, head, query_pos),
    // iterates K in [max(0, q-W+1), q] with online softmax.
    const size_t elemBytes = size_t(batch) * num_heads * seq_len * head_dim * 4;
    uint32_t outId = registry_.alloc(elemBytes);
    GrillyBuffer& qBuf = registry_.resolve(q_id);
    GrillyBuffer& kBuf = registry_.resolve(k_id);
    GrillyBuffer& vBuf = registry_.resolve(v_id);
    GrillyBuffer& outBuf = registry_.resolve(outId);

    constexpr uint32_t kNumBindings = 4;
    constexpr uint32_t kPushConstBytes = 6 * sizeof(uint32_t);  // 24 bytes
    PipelineEntry pipe = cache_.getOrCreate("chunked-sw-attention",
                                            kNumBindings, kPushConstBytes);
    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {qBuf.handle,   0, elemBytes},
        {kBuf.handle,   0, elemBytes},
        {vBuf.handle,   0, elemBytes},
        {outBuf.handle, 0, elemBytes},
    };
    VkDescriptorSet descSet =
        cache_.allocDescriptorSet("chunked-sw-attention", bufInfos);

    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    struct {
        uint32_t batch_size;
        uint32_t num_heads;
        uint32_t seq_len;
        uint32_t head_dim;
        uint32_t window_size;
        float scale;
    } push = {batch, num_heads, seq_len, head_dim, window_size, scale};
    static_assert(sizeof(push) == 24, "push constants must be 24 bytes");

    const uint32_t gx = batch * num_heads * seq_len;
    batch_.dispatch(pipe.pipeline, pipe.layout, descSet, gx, 1, 1,
                    &push, sizeof(push));
    batch_.transferComputeBarrier();
    return outId;
}

std::pair<uint32_t, uint32_t> TapeContext::fused_ce(
    uint32_t logits_id, uint32_t targets_id, uint32_t batch, uint32_t classes) {
    // Fused per-row cross-entropy: ONE on-chip dispatch -> per-row losses[batch] +
    // grad_logits[batch*classes] (softmax - one_hot), never materializing the full
    // softmax to VRAM and WITHOUT a host readback. Ignore-index: targets[row] >=
    // classes => that row's loss + grad are zero (prompt-masked / completion-only
    // SFT and RLVR pass masked positions this way). targets are uint32.
    const size_t lossBytes = size_t(batch) * 4;
    const size_t gradBytes = size_t(batch) * classes * 4;
    uint32_t lossesId = registry_.alloc(lossBytes);
    uint32_t gradId   = registry_.alloc(gradBytes);
    GrillyBuffer& logitsBuf  = registry_.resolve(logits_id);
    GrillyBuffer& targetsBuf = registry_.resolve(targets_id);
    GrillyBuffer& lossesBuf  = registry_.resolve(lossesId);
    GrillyBuffer& gradBuf    = registry_.resolve(gradId);

    constexpr uint32_t kNumBindings = 4;
    constexpr uint32_t kPushConstBytes = 2 * sizeof(uint32_t);
    PipelineEntry pipe = cache_.getOrCreate("loss-ce-fused", kNumBindings, kPushConstBytes);
    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {logitsBuf.handle,  0, gradBytes},
        {targetsBuf.handle, 0, lossBytes},
        {lossesBuf.handle,  0, lossBytes},
        {gradBuf.handle,    0, gradBytes},
    };
    VkDescriptorSet descSet = cache_.allocDescriptorSet("loss-ce-fused", bufInfos);
    struct { uint32_t batch_size; uint32_t num_classes; } push = {batch, classes};
    batch_.dispatch(pipe.pipeline, pipe.layout, descSet, batch, 1, 1, &push, sizeof(push));
    batch_.transferComputeBarrier();
    return {lossesId, gradId};
}

std::tuple<uint32_t, uint32_t, uint32_t>
TapeContext::forward_qkv_split(
    uint32_t qkv_id, uint32_t batch, uint32_t seq_len,
    uint32_t num_heads, uint32_t head_dim) {
    // Reshape fused (B*S, 3*H*Dh) QKV buffer into 3 separate (B, H, S, Dh)
    // buffers. attention-qkv-split.glsl: one thread per output element.
    const size_t outBytes = size_t(batch) * num_heads * seq_len * head_dim * 4;
    uint32_t qId = registry_.alloc(outBytes);
    uint32_t kId = registry_.alloc(outBytes);
    uint32_t vId = registry_.alloc(outBytes);
    GrillyBuffer& qkvBuf = registry_.resolve(qkv_id);
    GrillyBuffer& qBuf = registry_.resolve(qId);
    GrillyBuffer& kBuf = registry_.resolve(kId);
    GrillyBuffer& vBuf = registry_.resolve(vId);

    constexpr uint32_t kNumBindings = 4;
    constexpr uint32_t kPushConstBytes = 4 * sizeof(uint32_t);  // 16 bytes
    PipelineEntry pipe = cache_.getOrCreate("attention-qkv-split",
                                            kNumBindings, kPushConstBytes);
    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {qkvBuf.handle, 0, outBytes * 3},  // 3x the per-head size
        {qBuf.handle,   0, outBytes},
        {kBuf.handle,   0, outBytes},
        {vBuf.handle,   0, outBytes},
    };
    VkDescriptorSet descSet = cache_.allocDescriptorSet("attention-qkv-split", bufInfos);

    struct { uint32_t batch; uint32_t heads; uint32_t seq; uint32_t dh; } push =
        {batch, num_heads, seq_len, head_dim};
    const uint32_t total = batch * num_heads * seq_len * head_dim;
    batch_.dispatch(pipe.pipeline, pipe.layout, descSet,
                    (total + 63) / 64, 1, 1, &push, sizeof(push));
    batch_.transferComputeBarrier();
    return std::make_tuple(qId, kId, vId);
}

uint32_t TapeContext::forward_transpose_bhsd_bshd(
    uint32_t in_id, uint32_t batch, uint32_t num_heads,
    uint32_t seq_len, uint32_t head_dim) {
    // Transpose (B, H, S, Dh) attention output to (B*S, D) for output projection.
    // attention-transpose-bhsd-bshd.glsl: one thread per element.
    const size_t bytes = size_t(batch) * num_heads * seq_len * head_dim * 4;
    uint32_t outId = registry_.alloc(bytes);
    GrillyBuffer& inBuf = registry_.resolve(in_id);
    GrillyBuffer& outBuf = registry_.resolve(outId);

    constexpr uint32_t kNumBindings = 2;
    constexpr uint32_t kPushConstBytes = 4 * sizeof(uint32_t);
    PipelineEntry pipe = cache_.getOrCreate("attention-transpose-bhsd-bshd",
                                            kNumBindings, kPushConstBytes);
    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {inBuf.handle,  0, bytes},
        {outBuf.handle, 0, bytes},
    };
    VkDescriptorSet descSet = cache_.allocDescriptorSet("attention-transpose-bhsd-bshd", bufInfos);

    struct { uint32_t batch; uint32_t heads; uint32_t seq; uint32_t dh; } push =
        {batch, num_heads, seq_len, head_dim};
    const uint32_t total = batch * num_heads * seq_len * head_dim;
    batch_.dispatch(pipe.pipeline, pipe.layout, descSet,
                    (total + 63) / 64, 1, 1, &push, sizeof(push));
    batch_.transferComputeBarrier();
    return outId;
}

uint32_t TapeContext::forward_embedding(uint32_t ids_id, uint32_t table_id,
                                        uint32_t batch, uint32_t seqLen,
                                        uint32_t vocab, uint32_t dim) {
    // out[t,:] = table[ids[t]]; ids uint32 (batch*seq), table (vocab,dim).
    // embedding-lookup.glsl: one thread per token, copies dim floats.
    const uint32_t tokens = batch * seqLen;
    const size_t outBytes = size_t(tokens) * dim * 4;
    uint32_t outId = registry_.alloc(outBytes);
    GrillyBuffer& idsBuf = registry_.resolve(ids_id);
    GrillyBuffer& tblBuf = registry_.resolve(table_id);
    GrillyBuffer& outBuf = registry_.resolve(outId);

    PipelineEntry pipe =
        cache_.getOrCreate("embedding-lookup", 3, 4 * sizeof(uint32_t));
    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {idsBuf.handle, 0, size_t(tokens) * 4},
        {tblBuf.handle, 0, size_t(vocab) * dim * 4},
        {outBuf.handle, 0, outBytes},
    };
    VkDescriptorSet descSet =
        cache_.allocDescriptorSet("embedding-lookup", bufInfos);

    struct { uint32_t batch; uint32_t seqLen; uint32_t vocab; uint32_t dim; }
        push = {batch, seqLen, vocab, dim};
    uint32_t gx = (tokens + 255u) / 256u;
    batch_.dispatch(pipe.pipeline, pipe.layout, descSet, gx, 1, 1,
                    &push, sizeof(push));
    batch_.transferComputeBarrier();
    return outId;
}

uint32_t TapeContext::forward_add(uint32_t a_id, uint32_t b_id,
                                  uint32_t totalElements) {
    // out = a + b, NON-destructive (a and b survive for the backward tape).
    // Built from the verified fillZero + in-place batchedAdd primitives: zero
    // out, then out += a, out += b. Barriers order the transfer-write before the
    // first add and each read-modify-write of `out` before the next (the same
    // discipline the backward fan-out accumulation uses).
    const size_t bytes = size_t(totalElements) * 4;
    uint32_t outId = registry_.alloc(bytes);
    GrillyBuffer& outBuf = registry_.resolve(outId);
    GrillyBuffer& aBuf = registry_.resolve(a_id);
    GrillyBuffer& bBuf = registry_.resolve(b_id);

    batch_.fillZero(outBuf, bytes);
    batch_.transferComputeBarrier();
    grilly::ops::batchedAdd(batch_, cache_, outBuf, aBuf, totalElements);
    batch_.transferComputeBarrier();
    grilly::ops::batchedAdd(batch_, cache_, outBuf, bBuf, totalElements);
    batch_.transferComputeBarrier();
    return outId;
}

void TapeContext::adamw_update(uint32_t w_id, uint32_t grad_id, uint32_t m_id,
                               uint32_t v_id, uint32_t numel, float lr,
                               float beta1, float beta2, float eps,
                               float weight_decay, float beta1_t, float beta2_t,
                               bool clear_grad, float grad_scale) {
    // Dispatch adamw-update.glsl: 4 buffers {W, grad, m, v}, push matches the
    // shader's PushConsts exactly. In-place on W/m/v. grad_scale multiplies the
    // gradient BEFORE the m/v update (global grad-norm clip + mean-CE 1/B). No
    // begin/submit here -- the caller batches all updates into one submit.
    GrillyBuffer& wBuf = registry_.resolve(w_id);
    GrillyBuffer& gBuf = registry_.resolve(grad_id);
    GrillyBuffer& mBuf = registry_.resolve(m_id);
    GrillyBuffer& vBuf = registry_.resolve(v_id);
    const size_t bytes = size_t(numel) * 4;

    struct AdamWPush {
        uint32_t total_weights;
        float learning_rate, beta1, beta2, epsilon, weight_decay, beta1_t, beta2_t;
        uint32_t clear_grad;
        float grad_scale;
    } push = {numel, lr, beta1, beta2, eps, weight_decay, beta1_t, beta2_t,
              clear_grad ? 1u : 0u, grad_scale};

    PipelineEntry pipe = cache_.getOrCreate("adamw-update", 4, sizeof(push));
    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {wBuf.handle, 0, bytes},
        {gBuf.handle, 0, bytes},
        {mBuf.handle, 0, bytes},
        {vBuf.handle, 0, bytes},
    };
    VkDescriptorSet descSet = cache_.allocDescriptorSet("adamw-update", bufInfos);
    batch_.dispatch(pipe.pipeline, pipe.layout, descSet,
                    (numel + 255u) / 256u, 1, 1, &push, sizeof(push));
    batch_.transferComputeBarrier();
}

void TapeContext::embedding_scatter_add(uint32_t emb_grad_id, uint32_t ids_id,
                                        uint32_t e_grad_id, uint32_t tokens,
                                        uint32_t dim) {
    GrillyBuffer& gBuf = registry_.resolve(emb_grad_id);
    GrillyBuffer& idsBuf = registry_.resolve(ids_id);
    GrillyBuffer& egBuf = registry_.resolve(e_grad_id);
    const uint32_t total = tokens * dim;

    PipelineEntry pipe =
        cache_.getOrCreate("embedding-backward", 3, 2 * sizeof(uint32_t));
    std::vector<VkDescriptorBufferInfo> bufInfos = {
        {gBuf.handle,   0, size_t(total) * 4},
        {idsBuf.handle, 0, size_t(tokens) * 4},
        {egBuf.handle,  0, VK_WHOLE_SIZE},
    };
    VkDescriptorSet ds = cache_.allocDescriptorSet("embedding-backward", bufInfos);
    struct { uint32_t tokens; uint32_t dim; } push = {tokens, dim};
    batch_.begin();
    // Make prior writes to E_grad (the head-weight grad from backward, or an
    // upload) visible to this dispatch's atomicAdd read-modify-write.
    batch_.transferComputeBarrier();
    batch_.dispatch(pipe.pipeline, pipe.layout, ds, (total + 255u) / 256u, 1, 1,
                    &push, sizeof(push));
    batch_.transferComputeBarrier();
    batch_.submit();   // synchronous; E_grad now = head grad + embedding scatter
}

std::vector<float> TapeContext::bench_gemm(uint32_t M, uint32_t K, uint32_t N,
                                           uint32_t iters) {
    // Time gemm_tiled (fp32) vs gemm-coopmat-shared (fp16->fp32) for an MxKxN
    // GEMM, iters dispatches per submit (barrier between). Returns {fp32_ms,
    // fp16_ms} per iter. Values are garbage (timing only).
    const size_t mn32 = size_t(M) * N * 4;
    uint32_t a32 = registry_.alloc(size_t(M) * K * 4);
    uint32_t b32 = registry_.alloc(size_t(K) * N * 4);
    uint32_t c32 = registry_.alloc(mn32);
    uint32_t a16 = registry_.alloc(size_t(M) * K * 2);
    uint32_t b16 = registry_.alloc(size_t(K) * N * 2);
    uint32_t c16 = registry_.alloc(mn32);
    GrillyBuffer& A32 = registry_.resolve(a32); GrillyBuffer& B32 = registry_.resolve(b32);
    GrillyBuffer& C32 = registry_.resolve(c32);
    GrillyBuffer& A16 = registry_.resolve(a16); GrillyBuffer& B16 = registry_.resolve(b16);
    GrillyBuffer& C16 = registry_.resolve(c16);
    struct { uint32_t M, K, N, transpose_b; } pc = {M, K, N, 0u};
    // bench allocates B16 as a plain (K,N) row-major matrix, so transpose_b=0.
    struct { uint32_t M, K, N; } pc3 = {M, K, N};

    PipelineEntry tp = cache_.getOrCreate("gemm_tiled", 3, sizeof(pc3));
    std::vector<VkDescriptorBufferInfo> tb = {
        {A32.handle, 0, size_t(M) * K * 4}, {B32.handle, 0, size_t(K) * N * 4}, {C32.handle, 0, mn32}};
    VkDescriptorSet tds = cache_.allocDescriptorSet("gemm_tiled", tb);
    PipelineEntry cp = cache_.getOrCreate("gemm-coopmat-shared", 3, sizeof(pc));
    std::vector<VkDescriptorBufferInfo> cb = {
        {A16.handle, 0, size_t(M) * K * 2}, {B16.handle, 0, size_t(K) * N * 2}, {C16.handle, 0, mn32}};
    VkDescriptorSet cds = cache_.allocDescriptorSet("gemm-coopmat-shared", cb);

    auto timeIt = [&](VkPipeline pl, VkPipelineLayout ly, VkDescriptorSet ds,
                      uint32_t gx, uint32_t gy,
                      const void* push, uint32_t pushSize) {
        batch_.begin();
        for (uint32_t i = 0; i < iters; ++i) {
            batch_.dispatch(pl, ly, ds, gx, gy, 1, push, pushSize);
            batch_.barrier();
        }
        auto t0 = std::chrono::high_resolution_clock::now();
        batch_.submit();
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<float, std::milli>(t1 - t0).count() / iters;
    };
    // gemm_tiled gets the 12-byte {M,K,N} push; coopmat gets 16-byte {M,K,N,transpose_b}.
    timeIt(tp.pipeline, tp.layout, tds, (N + 63u) / 64u, (M + 63u) / 64u, &pc3, sizeof(pc3));  // warmup
    float fp32 = timeIt(tp.pipeline, tp.layout, tds, (N + 63u) / 64u, (M + 63u) / 64u, &pc3, sizeof(pc3));
    timeIt(cp.pipeline, cp.layout, cds, N / 64u, M / 16u, &pc, sizeof(pc));                   // warmup
    float fp16 = timeIt(cp.pipeline, cp.layout, cds, N / 64u, M / 16u, &pc, sizeof(pc));
    return {fp32, fp16};
}

float TapeContext::sum_squares(const std::vector<uint32_t>& ids,
                               const std::vector<uint32_t>& numels) {
    // acc[0] += sum x^2 over each buffer (reduce-sumsq.glsl, atomic accumulate).
    // One batch, one 4-byte readback. Each dispatch atomic-adds to the same acc,
    // so no inter-dispatch barrier is needed (atomics serialize the writes).
    uint32_t accId = registry_.alloc(4, BufferRegistry::Kind::Readback);
    GrillyBuffer& accBuf = registry_.resolve(accId);

    batch_.begin();
    batch_.fillZero(accBuf, 4);
    batch_.transferComputeBarrier();
    PipelineEntry pipe = cache_.getOrCreate("reduce-sumsq", 2, sizeof(uint32_t));
    for (size_t i = 0; i < ids.size(); ++i) {
        GrillyBuffer& inBuf = registry_.resolve(ids[i]);
        uint32_t n = numels[i];
        std::vector<VkDescriptorBufferInfo> bufInfos = {
            {inBuf.handle,  0, size_t(n) * 4},
            {accBuf.handle, 0, 4},
        };
        VkDescriptorSet ds = cache_.allocDescriptorSet("reduce-sumsq", bufInfos);
        batch_.dispatch(pipe.pipeline, pipe.layout, ds, (n + 255u) / 256u, 1, 1,
                        &n, sizeof(n));
    }
    batch_.transferComputeBarrier();
    batch_.submit();   // synchronous: acc ready

    float result = 0.0f;
    registry_.download(accId, &result, 4);
    return result;
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
