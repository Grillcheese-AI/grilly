#include "grilly/nn/snn.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace grilly {
namespace nn {

// ══════════════════════════════════════════════════════════════════════
// MemoryModule
// ══════════════════════════════════════════════════════════════════════

void MemoryModule::register_memory(const std::string& name,
                                   Tensor default_value) {
    MemoryVar mv;
    mv.name = name;
    mv.default_value = default_value;
    // Initialize current to a copy of default
    auto shape = default_value.shape();
    mv.current = Tensor::zeros(shape, default_value.backend());
    memories_.push_back(std::move(mv));
}

void MemoryModule::reset() {
    for (auto& mv : memories_) {
        // Reset to default value (zeros for typical SNN use)
        auto shape = mv.default_value.shape();
        mv.current = Tensor::zeros(shape, mv.default_value.backend());
    }
    // Recurse into child modules
    for (auto& [name, mod] : modules_) {
        auto* mem_mod = dynamic_cast<MemoryModule*>(mod.get());
        if (mem_mod) mem_mod->reset();
    }
}

void MemoryModule::detach() {
    for (auto& mv : memories_) {
        mv.current.set_requires_grad(false);
        mv.current.set_grad(nullptr);
    }
    for (auto& [name, mod] : modules_) {
        auto* mem_mod = dynamic_cast<MemoryModule*>(mod.get());
        if (mem_mod) mem_mod->detach();
    }
}

Tensor* MemoryModule::get_memory(const std::string& name) {
    for (auto& mv : memories_) {
        if (mv.name == name) return &mv.current;
    }
    return nullptr;
}

const Tensor* MemoryModule::get_memory(const std::string& name) const {
    for (const auto& mv : memories_) {
        if (mv.name == name) return &mv.current;
    }
    return nullptr;
}

// ══════════════════════════════════════════════════════════════════════
// BaseNode
// ══════════════════════════════════════════════════════════════════════

BaseNode::BaseNode(float v_threshold, float v_reset,
                   SurrogateFunction surrogate, bool detach_reset,
                   std::string step_mode, bool store_v_seq)
    : v_threshold_(v_threshold),
      v_reset_(v_reset),
      surrogate_(surrogate),
      detach_reset_(detach_reset),
      step_mode_(std::move(step_mode)),
      store_v_seq_(store_v_seq) {
    // Register membrane potential as a memory variable
    // Shape will be set dynamically on first forward pass
    register_memory("v", Tensor({0}, DType::Float32, nullptr));
}

Tensor BaseNode::neuronal_fire() {
    // h = v - v_threshold (pre-spike potential)
    Tensor* v = get_memory("v");
    if (!v) throw std::runtime_error("BaseNode: membrane potential 'v' not found");

    int64_t n = v->numel();
    const float* v_data = v->data();
    std::vector<float> h_data(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; i++) {
        h_data[i] = v_data[i] - v_threshold_;
    }

    Tensor h(std::move(h_data), v->shape(), v->backend());

    // Apply surrogate forward (Heaviside) to get spike
    return surrogate_.forward(h);
}

void BaseNode::neuronal_reset(const Tensor& spike) {
    Tensor* v = get_memory("v");
    if (!v) return;

    int64_t n = v->numel();
    float* v_data = v->mutable_data();
    const float* spike_data = spike.data();

    if (has_reset_) {
        // Hard reset: v = v * (1 - spike) + v_reset * spike
        for (int64_t i = 0; i < n; i++) {
            v_data[i] = v_data[i] * (1.0f - spike_data[i]) +
                        v_reset_ * spike_data[i];
        }
    } else {
        // Soft reset: v = v - v_threshold * spike
        for (int64_t i = 0; i < n; i++) {
            v_data[i] -= v_threshold_ * spike_data[i];
        }
    }
}

Tensor BaseNode::forward(Tensor input) {
    if (step_mode_ == "m") {
        return multi_step_forward(input);
    }
    return single_step_forward(input);
}

Tensor BaseNode::single_step_forward(const Tensor& x) {
    // Initialize membrane potential if shape doesn't match
    Tensor* v = get_memory("v");
    if (!v || v->numel() != x.numel()) {
        // Re-register with correct shape
        memories_.clear();
        register_memory("v", Tensor::zeros(x.shape(), x.backend()));
        v = get_memory("v");
    }

    // 1. Charge: accumulate input into membrane
    neuronal_charge(x);

    // 2. Fire: threshold crossing → spike
    Tensor spike = neuronal_fire();

    // 3. Reset: reduce membrane where spike occurred
    neuronal_reset(spike);

    return spike;
}

Tensor BaseNode::multi_step_forward(const Tensor& x_seq) {
    // x_seq shape: [T, N, ...] where T = timesteps
    if (x_seq.ndim() < 2) {
        throw std::invalid_argument(
            "BaseNode::multi_step_forward: input must be at least 2D [T, ...]");
    }

    int64_t T = x_seq.shape(0);
    auto step_shape = std::vector<int64_t>(x_seq.shape().begin() + 1,
                                           x_seq.shape().end());
    int64_t step_numel = std::accumulate(step_shape.begin(), step_shape.end(),
                                         int64_t(1), std::multiplies<int64_t>());

    // Try GPU-fused path if backend is available
    ComputeBackend* backend = x_seq.backend();
    if (!backend) backend = backend_;

    if (backend && backend->hasShader("snn-node-forward")) {
        return multi_step_forward_gpu(x_seq, backend, T, step_shape, step_numel);
    }

    // CPU fallback: loop over timesteps
    const float* x_data = x_seq.data();
    std::vector<float> all_spikes(static_cast<size_t>(T * step_numel));

    // Clear backward caches
    h_seq_cache_.clear();
    spike_seq_cache_.clear();

    for (int64_t t = 0; t < T; t++) {
        // Extract timestep slice
        std::vector<float> step_data(x_data + t * step_numel,
                                     x_data + (t + 1) * step_numel);
        Tensor x_t(std::move(step_data), step_shape, backend);

        // Single step
        Tensor spike_t = single_step_forward(x_t);

        // Cache for backward
        if (training_) {
            Tensor* v = get_memory("v");
            if (v) {
                // Cache h = v - v_threshold (before spike)
                std::vector<float> h_data(static_cast<size_t>(step_numel));
                const float* v_data = v->data();
                for (int64_t i = 0; i < step_numel; i++) {
                    h_data[i] = v_data[i] - v_threshold_;
                }
                h_seq_cache_.emplace_back(std::move(h_data), step_shape, backend);
            }
            spike_seq_cache_.push_back(spike_t);
        }

        // Copy spike into output
        const float* spike_ptr = spike_t.data();
        std::memcpy(all_spikes.data() + t * step_numel, spike_ptr,
                    step_numel * sizeof(float));
    }

    // Build output shape [T, N, ...]
    std::vector<int64_t> out_shape = {T};
    out_shape.insert(out_shape.end(), step_shape.begin(), step_shape.end());

    return Tensor(std::move(all_spikes), std::move(out_shape), backend);
}

Tensor BaseNode::backward(const Tensor& grad_output) {
    // BPTT with surrogate gradients
    // grad_output shape: [T, N, ...] matching multi_step_forward output
    if (h_seq_cache_.empty()) {
        throw std::runtime_error(
            "BaseNode::backward: no cached forward pass data. "
            "Run forward in training mode first.");
    }

    int64_t T = static_cast<int64_t>(h_seq_cache_.size());
    auto step_shape = h_seq_cache_[0].shape();
    int64_t step_numel = h_seq_cache_[0].numel();

    const float* grad_data = grad_output.data();
    std::vector<float> grad_input(static_cast<size_t>(T * step_numel), 0.0f);

    // Backward through time
    for (int64_t t = T - 1; t >= 0; t--) {
        // Surrogate gradient: dspike/dh ≈ surrogate.gradient(h)
        Tensor surrogate_grad = surrogate_.gradient(h_seq_cache_[t]);
        const float* sg = surrogate_grad.data();
        const float* go = grad_data + t * step_numel;

        // grad_input[t] = grad_output[t] * surrogate_grad(h[t])
        for (int64_t i = 0; i < step_numel; i++) {
            grad_input[t * step_numel + i] = go[i] * sg[i];
        }
    }

    std::vector<int64_t> out_shape = {T};
    out_shape.insert(out_shape.end(), step_shape.begin(), step_shape.end());

    return Tensor(std::move(grad_input), std::move(out_shape),
                  grad_output.backend());
}

// ── GPU-fused multi-step (private helper) ─────────────────────────────

Tensor BaseNode::multi_step_forward_gpu(const Tensor& x_seq,
                                        ComputeBackend* backend,
                                        int64_t T,
                                        const std::vector<int64_t>& step_shape,
                                        int64_t step_numel) {
    size_t step_bytes = static_cast<size_t>(step_numel) * sizeof(float);

    // Ensure input is on GPU
    Tensor x_gpu = x_seq;
    if (!x_gpu.on_gpu()) {
        x_gpu.set_backend(backend);
        x_gpu.ensure_gpu();
    }

    // Allocate membrane potential on GPU
    Tensor* v = get_memory("v");
    if (!v || v->numel() != step_numel) {
        memories_.clear();
        register_memory("v", Tensor::zeros(
            std::vector<int64_t>(step_shape), backend));
        v = get_memory("v");
    }
    v->set_backend(backend);
    v->ensure_gpu();

    // Allocate output spike buffers on GPU
    std::vector<uint64_t> spike_handles(T);
    for (int64_t t = 0; t < T; t++) {
        BufferDesc desc;
        desc.size = step_bytes;
        desc.usage = BufferDesc::DeviceLocal;
        spike_handles[t] = backend->createBuffer(desc);
    }

    // Allocate h buffer (pre-spike voltage, for backward caching)
    BufferDesc h_desc;
    h_desc.size = step_bytes;
    h_desc.usage = BufferDesc::DeviceLocal;
    uint64_t h_handle = backend->createBuffer(h_desc);

    // Push constants for the SNN shader
    struct SNNPushConstants {
        uint32_t num_elements;
        float v_threshold;
        float v_reset;
        float tau;
        uint32_t neuron_type;  // 0=IF, 1=LIF, 2=PLIF
        uint32_t decay_input;  // bool
        uint32_t has_reset;    // bool (hard vs soft)
        uint32_t surrogate_type;
        float surrogate_alpha;
        uint32_t _pad[3];      // Alignment to 48 bytes
    };

    SNNPushConstants push{};
    push.num_elements = static_cast<uint32_t>(step_numel);
    push.v_threshold = v_threshold_;
    push.v_reset = v_reset_;
    push.tau = gpu_tau();
    push.neuron_type = gpu_neuron_type();
    push.decay_input = gpu_decay_input() ? 1u : 0u;
    push.has_reset = has_reset_ ? 1u : 0u;
    push.surrogate_type = static_cast<uint32_t>(surrogate_.type);
    push.surrogate_alpha = surrogate_.alpha;

    uint32_t workgroups = (static_cast<uint32_t>(step_numel) + 255) / 256;
    uint64_t v_handle = v->gpu_handle();

    // ★ Record all T timesteps into a single command batch
    backend->beginBatch();
    for (int64_t t = 0; t < T; t++) {
        // x_seq is contiguous [T, step_numel], slice by offset
        // We pass the full x buffer + an offset via push constants
        // For simplicity, we use separate dispatches with barriers
        uint64_t x_offset_handle = x_gpu.gpu_handle();  // Full buffer

        // Update push constant with timestep offset
        struct SNNStepPush {
            uint32_t num_elements;
            float v_threshold;
            float v_reset;
            float tau;
            uint32_t neuron_type;
            uint32_t decay_input;
            uint32_t has_reset;
            uint32_t surrogate_type;
            float surrogate_alpha;
            uint32_t timestep_offset;
            uint32_t _pad[2];
        } step_push{};

        step_push.num_elements = static_cast<uint32_t>(step_numel);
        step_push.v_threshold = v_threshold_;
        step_push.v_reset = v_reset_;
        step_push.tau = gpu_tau();
        step_push.neuron_type = gpu_neuron_type();
        step_push.decay_input = gpu_decay_input() ? 1u : 0u;
        step_push.has_reset = has_reset_ ? 1u : 0u;
        step_push.surrogate_type = static_cast<uint32_t>(surrogate_.type);
        step_push.surrogate_alpha = surrogate_.alpha;
        step_push.timestep_offset = static_cast<uint32_t>(t);

        backend->dispatch(
            "snn-node-forward",
            {x_offset_handle, v_handle, spike_handles[t], h_handle},
            workgroups, 1, 1,
            &step_push, sizeof(step_push));

        // Memory barrier: V[t] must complete before V[t+1]
        backend->barrier();
    }
    backend->endBatch();  // ★ Single fence wait for all T steps!

    // Mark membrane potential as GPU-modified
    v->mark_gpu_modified();

    // Download and stack spike results
    std::vector<float> all_spikes(static_cast<size_t>(T * step_numel));
    for (int64_t t = 0; t < T; t++) {
        backend->download(spike_handles[t],
                          all_spikes.data() + t * step_numel,
                          step_bytes);
        backend->destroyBuffer(spike_handles[t]);
    }
    backend->destroyBuffer(h_handle);

    std::vector<int64_t> out_shape = {T};
    out_shape.insert(out_shape.end(), step_shape.begin(), step_shape.end());

    return Tensor(std::move(all_spikes), std::move(out_shape), backend);
}

// ══════════════════════════════════════════════════════════════════════
// IFNode
// ══════════════════════════════════════════════════════════════════════

IFNode::IFNode(float v_threshold, float v_reset,
               SurrogateFunction surrogate, bool detach_reset,
               std::string step_mode, bool store_v_seq)
    : BaseNode(v_threshold, v_reset, std::move(surrogate), detach_reset,
               std::move(step_mode), store_v_seq) {}

void IFNode::neuronal_charge(const Tensor& x) {
    // V[t] = V[t-1] + X[t]
    Tensor* v = get_memory("v");
    if (!v) return;

    int64_t n = x.numel();
    float* v_data = v->mutable_data();
    const float* x_data = x.data();

    for (int64_t i = 0; i < n; i++) {
        v_data[i] += x_data[i];
    }
}

// ══════════════════════════════════════════════════════════════════════
// LIFNode
// ══════════════════════════════════════════════════════════════════════

LIFNode::LIFNode(float tau, bool decay_input,
                 float v_threshold, float v_reset,
                 SurrogateFunction surrogate, bool detach_reset,
                 std::string step_mode, bool store_v_seq)
    : BaseNode(v_threshold, v_reset, std::move(surrogate), detach_reset,
               std::move(step_mode), store_v_seq),
      tau_(tau),
      decay_input_(decay_input) {
    if (tau <= 0.0f) {
        throw std::invalid_argument("LIFNode: tau must be positive");
    }
}

void LIFNode::neuronal_charge(const Tensor& x) {
    Tensor* v = get_memory("v");
    if (!v) return;

    int64_t n = x.numel();
    float* v_data = v->mutable_data();
    const float* x_data = x.data();
    float decay = 1.0f / tau_;

    if (decay_input_) {
        // V[t] = decay * (V[t-1] + X[t])
        for (int64_t i = 0; i < n; i++) {
            v_data[i] = decay * (v_data[i] + x_data[i]);
        }
    } else {
        // V[t] = decay * V[t-1] + X[t]
        for (int64_t i = 0; i < n; i++) {
            v_data[i] = decay * v_data[i] + x_data[i];
        }
    }
}

// ══════════════════════════════════════════════════════════════════════
// ParametricLIFNode
// ══════════════════════════════════════════════════════════════════════

ParametricLIFNode::ParametricLIFNode(float init_tau, bool decay_input,
                                     float v_threshold, float v_reset,
                                     SurrogateFunction surrogate,
                                     bool detach_reset,
                                     std::string step_mode,
                                     bool store_v_seq)
    : BaseNode(v_threshold, v_reset, std::move(surrogate), detach_reset,
               std::move(step_mode), store_v_seq),
      decay_input_(decay_input) {
    if (init_tau <= 0.0f) {
        throw std::invalid_argument("ParametricLIFNode: init_tau must be positive");
    }
    // Register tau as a learnable parameter
    std::vector<float> tau_data = {init_tau};
    Parameter tau_param(Tensor(std::move(tau_data), {1}, nullptr), true);
    register_parameter("tau", std::move(tau_param));
}

void ParametricLIFNode::neuronal_charge(const Tensor& x) {
    Tensor* v = get_memory("v");
    if (!v) return;

    // Get learnable tau
    auto it = parameters_.find("tau");
    if (it == parameters_.end()) {
        throw std::runtime_error("ParametricLIFNode: 'tau' parameter not found");
    }
    float tau_val = it->second.data()[0];
    float decay = 1.0f / tau_val;

    int64_t n = x.numel();
    float* v_data = v->mutable_data();
    const float* x_data = x.data();

    if (decay_input_) {
        for (int64_t i = 0; i < n; i++) {
            v_data[i] = decay * (v_data[i] + x_data[i]);
        }
    } else {
        for (int64_t i = 0; i < n; i++) {
            v_data[i] = decay * v_data[i] + x_data[i];
        }
    }
}

float ParametricLIFNode::gpu_tau() const {
    auto it = parameters_.find("tau");
    if (it == parameters_.end()) return 2.0f;
    return it->second.data()[0];
}

}  // namespace nn
}  // namespace grilly
