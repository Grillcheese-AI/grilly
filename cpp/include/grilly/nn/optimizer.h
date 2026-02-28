#pragma once
/// Optimizers with batched GPU dispatch.
///
/// The key optimization: all parameter updates are recorded into a single
/// CommandBatch and submitted with one fence wait. The Python optimizer
/// does N separate dispatches with N fence waits — this batches them.
///
/// Supports: Adam, AdamW, SGD.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "grilly/compute_backend.h"
#include "grilly/nn/parameter.h"
#include "grilly/nn/tensor.h"

namespace py = pybind11;

namespace grilly {
namespace nn {

/// Base optimizer class with parameter groups and per-parameter state.
class Optimizer {
public:
    virtual ~Optimizer() = default;

    /// Perform one optimization step (batched GPU dispatch).
    virtual void step() = 0;

    /// Zero all gradients across all parameter groups.
    void zero_grad();

    /// Serialization for checkpointing.
    py::dict state_dict() const;
    void load_state_dict(py::dict state);

    /// Number of parameters managed.
    size_t param_count() const;

protected:
    struct ParamGroup {
        std::vector<Parameter*> params;
        std::map<std::string, float> options;
    };
    std::vector<ParamGroup> param_groups_;

    /// Refresh parameter pointers from Python list (pybind11 safety).
    void refresh_params();

    /// Per-parameter optimizer state (exp_avg, exp_avg_sq, etc.)
    struct ParamState {
        Tensor exp_avg;
        Tensor exp_avg_sq;
        Tensor max_exp_avg_sq;  // For AMSGrad
        int step_count = 0;
    };
    std::unordered_map<size_t, ParamState> state_;  // Key = param index

    ComputeBackend* backend_ = nullptr;

public:
    /// Keep Python param objects alive to prevent GC
    py::list py_params_;
};

/// Adam optimizer with batched GPU dispatch.
class Adam : public Optimizer {
public:
    Adam(std::vector<Parameter*> params,
         float lr = 1e-3f,
         std::pair<float, float> betas = {0.9f, 0.999f},
         float eps = 1e-8f,
         float weight_decay = 0.0f,
         bool amsgrad = false);

    void step() override;

    float lr() const { return lr_; }
    void set_lr(float lr) { lr_ = lr; }

private:
    float lr_;
    float beta1_, beta2_;
    float eps_;
    float weight_decay_;
    bool amsgrad_;
    int global_step_ = 0;
};

/// AdamW optimizer (decoupled weight decay).
class AdamW : public Optimizer {
public:
    AdamW(std::vector<Parameter*> params,
          float lr = 1e-3f,
          std::pair<float, float> betas = {0.9f, 0.999f},
          float eps = 1e-8f,
          float weight_decay = 0.01f,
          bool amsgrad = false);

    void step() override;

    float lr() const { return lr_; }
    void set_lr(float lr) { lr_ = lr; }

private:
    float lr_;
    float beta1_, beta2_;
    float eps_;
    float weight_decay_;
    bool amsgrad_;
    int global_step_ = 0;
};

/// SGD with optional momentum.
class SGD : public Optimizer {
public:
    SGD(std::vector<Parameter*> params,
        float lr = 0.01f,
        float momentum = 0.0f,
        float weight_decay = 0.0f,
        bool nesterov = false);

    void step() override;

    float lr() const { return lr_; }
    void set_lr(float lr) { lr_ = lr; }

private:
    float lr_;
    float momentum_;
    float weight_decay_;
    bool nesterov_;
};

}  // namespace nn
}  // namespace grilly
