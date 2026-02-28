#pragma once
/// Spiking Neural Network (SNN) engine — GPU-fused temporal loop.
///
/// Class hierarchy:
///   Module → MemoryModule → BaseNode → { IFNode, LIFNode, ParametricLIFNode }
///
/// MemoryModule adds stateful memory variables (membrane potential, etc.)
/// that persist across timesteps and can be reset between sequences.
///
/// BaseNode implements the core SNN dynamics:
///   1. neuronal_charge(x)  — subclass-specific: accumulate input into membrane
///   2. neuronal_fire()     — Heaviside threshold crossing → spike
///   3. neuronal_reset()    — hard/soft reset after spike
///
/// The key optimization is multi_step_forward(): the Python version loops
/// T timesteps with T separate GPU dispatches and numpy copies. The C++
/// version records all T dispatches into a single CommandBatch with
/// barriers between timesteps — 1 fence wait instead of T.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "grilly/nn/module.h"
#include "grilly/nn/parameter.h"
#include "grilly/nn/surrogate.h"
#include "grilly/nn/tensor.h"

namespace grilly {
namespace nn {

// ── MemoryModule ──────────────────────────────────────────────────────

/// Module with persistent state variables that survive across timesteps.
/// Used by SNN neurons (membrane potential) and recurrent cells.
class MemoryModule : public Module {
protected:
    struct MemoryVar {
        std::string name;
        Tensor default_value;  // Reset target
        Tensor current;        // Live state
    };
    std::vector<MemoryVar> memories_;

public:
    /// Register a named memory variable with a default (reset) value.
    void register_memory(const std::string& name, Tensor default_value);

    /// Reset all memories to their defaults (recursive through children).
    void reset();

    /// Detach all memories from the computation graph (break BPTT flow).
    void detach();

    /// Get a memory variable by name.
    Tensor* get_memory(const std::string& name);
    const Tensor* get_memory(const std::string& name) const;
};

// ── BaseNode ──────────────────────────────────────────────────────────

/// Base class for all spiking neuron models.
///
/// Subclasses override neuronal_charge() to define the membrane dynamics.
/// BaseNode handles fire/reset/multi-step orchestration.
class BaseNode : public MemoryModule {
public:
    BaseNode(float v_threshold = 1.0f,
             float v_reset = 0.0f,
             SurrogateFunction surrogate = ATan(),
             bool detach_reset = false,
             std::string step_mode = "s",
             bool store_v_seq = false);

    // ── Core dynamics (override neuronal_charge in subclasses) ──

    /// Accumulate input into membrane potential. Subclass-specific.
    virtual void neuronal_charge(const Tensor& x) = 0;

    /// Fire: threshold crossing → spike via Heaviside + surrogate.
    Tensor neuronal_fire();

    /// Reset: hard (v = v_reset) or soft (v = v - v_threshold * spike).
    void neuronal_reset(const Tensor& spike);

    // ── Forward dispatch ──

    Tensor forward(Tensor input) override;
    Tensor single_step_forward(const Tensor& x);

    /// GPU-fused multi-step: entire T-loop in one command buffer.
    Tensor multi_step_forward(const Tensor& x_seq);

    /// BPTT backward with surrogate gradients.
    Tensor backward(const Tensor& grad_output);

    // ── Accessors ──

    float v_threshold() const { return v_threshold_; }
    float v_reset() const { return v_reset_; }
    const std::string& step_mode() const { return step_mode_; }
    void set_step_mode(const std::string& mode) { step_mode_ = mode; }

    /// Whether v_reset is used (true = hard reset, false = soft reset).
    bool has_reset() const { return has_reset_; }
    void set_has_reset(bool hr) { has_reset_ = hr; }

protected:
    float v_threshold_;
    float v_reset_;
    bool has_reset_ = true;        // true = hard reset, false = soft reset
    SurrogateFunction surrogate_;
    bool detach_reset_;
    std::string step_mode_;        // "s" = single, "m" = multi
    bool store_v_seq_;

    // GPU neuron type for shader dispatch (subclasses set this)
    virtual uint8_t gpu_neuron_type() const = 0;
    virtual float gpu_tau() const { return 2.0f; }
    virtual bool gpu_decay_input() const { return false; }

    // Backward caches
    std::vector<Tensor> h_seq_cache_;
    std::vector<Tensor> spike_seq_cache_;

private:
    /// GPU-fused multi-step implementation.
    Tensor multi_step_forward_gpu(const Tensor& x_seq,
                                  ComputeBackend* backend,
                                  int64_t T,
                                  const std::vector<int64_t>& step_shape,
                                  int64_t step_numel);
};

// ── Concrete neurons ──────────────────────────────────────────────────

/// Integrate-and-Fire neuron. Simplest model: V[t] = V[t-1] + X[t].
class IFNode : public BaseNode {
public:
    IFNode(float v_threshold = 1.0f,
           float v_reset = 0.0f,
           SurrogateFunction surrogate = ATan(),
           bool detach_reset = false,
           std::string step_mode = "s",
           bool store_v_seq = false);

    void neuronal_charge(const Tensor& x) override;
    uint8_t gpu_neuron_type() const override { return 0; }
};

/// Leaky Integrate-and-Fire neuron.
/// V[t] = (1/τ) * V[t-1] + X[t]              (decay_input = false)
/// V[t] = (1/τ) * (V[t-1] + X[t])            (decay_input = true)
class LIFNode : public BaseNode {
public:
    LIFNode(float tau = 2.0f,
            bool decay_input = false,
            float v_threshold = 1.0f,
            float v_reset = 0.0f,
            SurrogateFunction surrogate = ATan(),
            bool detach_reset = false,
            std::string step_mode = "s",
            bool store_v_seq = false);

    void neuronal_charge(const Tensor& x) override;
    uint8_t gpu_neuron_type() const override { return 1; }
    float gpu_tau() const override { return tau_; }
    bool gpu_decay_input() const override { return decay_input_; }

    float tau() const { return tau_; }

private:
    float tau_;
    bool decay_input_;
};

/// Parametric LIF — tau is learnable (registered as a Parameter).
class ParametricLIFNode : public BaseNode {
public:
    ParametricLIFNode(float init_tau = 2.0f,
                      bool decay_input = false,
                      float v_threshold = 1.0f,
                      float v_reset = 0.0f,
                      SurrogateFunction surrogate = ATan(),
                      bool detach_reset = false,
                      std::string step_mode = "s",
                      bool store_v_seq = false);

    void neuronal_charge(const Tensor& x) override;
    uint8_t gpu_neuron_type() const override { return 2; }
    float gpu_tau() const override;
    bool gpu_decay_input() const override { return decay_input_; }

private:
    bool decay_input_;
    // tau_ is stored as a Parameter via register_parameter("tau", ...)
};

}  // namespace nn
}  // namespace grilly
