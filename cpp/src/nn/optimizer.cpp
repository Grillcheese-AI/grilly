#include "grilly/nn/optimizer.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace grilly {
namespace nn {

// ══════════════════════════════════════════════════════════════════════
// Optimizer base
// ══════════════════════════════════════════════════════════════════════

void Optimizer::refresh_params() {
    if (py_params_.size() > 0 && !param_groups_.empty()) {
        auto& params = param_groups_[0].params;
        params.clear();
        for (auto& item : py_params_) {
            params.push_back(item.cast<Parameter*>());
        }
        // Also refresh backend from first param
        if (!params.empty() && params[0]) {
            backend_ = params[0]->backend();
        }
    }
}

void Optimizer::zero_grad() {
    refresh_params();
    for (auto& group : param_groups_) {
        for (auto* p : group.params) {
            p->zero_grad();
        }
    }
}

size_t Optimizer::param_count() const {
    return py_params_.size();
}

py::dict Optimizer::state_dict() const {
    py::dict state;
    py::list param_states;

    std::vector<uintptr_t> keys;
    keys.reserve(state_.size());
    for (const auto& [key, _] : state_) {
        keys.push_back(key);
    }
    std::sort(keys.begin(), keys.end());

    for (uintptr_t key : keys) {
        const auto& pstate = state_.at(key);
        py::dict ps;
        ps["param_key"] = py::int_(static_cast<unsigned long long>(key));
        ps["step_count"] = pstate.step_count;
        if (pstate.exp_avg.numel() > 0) {
            ps["exp_avg"] = pstate.exp_avg.numpy();
        }
        if (pstate.exp_avg_sq.numel() > 0) {
            ps["exp_avg_sq"] = pstate.exp_avg_sq.numpy();
        }
        param_states.append(ps);
    }

    state["param_states"] = param_states;
    return state;
}

void Optimizer::load_state_dict(py::dict state) {
    (void)state;
}

// ══════════════════════════════════════════════════════════════════════
// Adam
// ══════════════════════════════════════════════════════════════════════

Adam::Adam(std::vector<Parameter*> params, float lr,
           std::pair<float, float> betas, float eps,
           float weight_decay, bool amsgrad)
    : lr_(lr),
      beta1_(betas.first),
      beta2_(betas.second),
      eps_(eps),
      weight_decay_(weight_decay),
      amsgrad_(amsgrad) {
    ParamGroup group;
    group.params = std::move(params);
    param_groups_.push_back(std::move(group));
}

void Adam::step() {
    refresh_params();
    global_step_++;

    float bc1 = 1.0f - std::pow(beta1_, static_cast<float>(global_step_));
    float bc2 = 1.0f - std::pow(beta2_, static_cast<float>(global_step_));

    // GPU-batched path
    bool has_gpu = backend_ && backend_->hasShader("adam-update");
    if (has_gpu) {
        // Must match adam-update.glsl push_constant layout exactly:
        //   uint total_weights, float lr, float beta1, float beta2,
        //   float epsilon, float beta1_t, float beta2_t, uint clear_grad
        struct AdamPush {
            uint32_t total_weights;
            float learning_rate;
            float beta1;
            float beta2;
            float epsilon;
            float beta1_t;      // beta1^t (raw power, NOT bias correction)
            float beta2_t;      // beta2^t (raw power, NOT bias correction)
            uint32_t clear_grad; // 1 = zero gradients after update
        };

        float beta1_t = std::pow(beta1_, static_cast<float>(global_step_));
        float beta2_t = std::pow(beta2_, static_cast<float>(global_step_));

        // Apply weight decay on CPU before GPU dispatch (L2 regularization)
        if (weight_decay_ > 0.0f) {
            for (auto& group : param_groups_) {
                for (auto* p : group.params) {
                    if (!p->has_grad()) continue;
                    // Add weight_decay * param to gradient (L2 reg)
                    int64_t n = p->numel();
                    float* gd = p->grad_ref().mutable_data();
                    const float* pd = p->data();
                    for (int64_t i = 0; i < n; i++) {
                        gd[i] += weight_decay_ * pd[i];
                    }
                }
            }
        }

        backend_->beginBatch();

        for (auto& group : param_groups_) {
            for (auto* p : group.params) {
                if (!p || !p->has_grad()) { continue; }

                uintptr_t key = reinterpret_cast<uintptr_t>(p);
                auto& s = state_[key];
                int64_t n = p->numel();

                if (s.exp_avg.numel() == 0) {
                    s.exp_avg = Tensor::zeros(p->shape(), backend_);
                    s.exp_avg.ensure_gpu();
                    s.exp_avg_sq = Tensor::zeros(p->shape(), backend_);
                    s.exp_avg_sq.ensure_gpu();
                }
                s.step_count = global_step_;

                uint32_t workgroups = (static_cast<uint32_t>(n) + 255) / 256;

                AdamPush push{};
                push.total_weights = static_cast<uint32_t>(n);
                push.learning_rate = lr_;
                push.beta1 = beta1_;
                push.beta2 = beta2_;
                push.epsilon = eps_;
                push.beta1_t = beta1_t;
                push.beta2_t = beta2_t;
                push.clear_grad = 0;  // Don't auto-clear; zero_grad() handles it

                // Propagate backend to param/grad if missing
                if (!p->backend()) p->set_backend(backend_);
                p->ensure_gpu();
                if (!p->grad_ref().backend())
                    p->grad_ref().set_backend(backend_);
                p->grad_ref().ensure_gpu();

                backend_->dispatch(
                    "adam-update",
                    {p->gpu_handle(), p->grad_ref().gpu_handle(),
                     s.exp_avg.gpu_handle(), s.exp_avg_sq.gpu_handle()},
                    workgroups, 1, 1,
                    &push, sizeof(push));
            }
        }

        backend_->endBatch();
        backend_->barrier();

        // Mark all parameters as GPU-modified
        for (auto& group : param_groups_) {
            for (auto* p : group.params) {
                if (p->has_grad()) {
                    p->mark_gpu_modified();
                }
            }
        }
        return;
    }

    // CPU fallback
    for (auto& group : param_groups_) {
        for (auto* p : group.params) {
            if (!p || !p->has_grad()) { continue; }

            uintptr_t key = reinterpret_cast<uintptr_t>(p);
            auto& s = state_[key];
            int64_t n = p->numel();

            if (s.exp_avg.numel() == 0) {
                s.exp_avg = Tensor::zeros(p->shape(), p->backend());
                s.exp_avg_sq = Tensor::zeros(p->shape(), p->backend());
            }
            s.step_count = global_step_;

            float* param_data = p->mutable_data();
            const float* grad_data = p->grad_ref().data();
            float* m_data = s.exp_avg.mutable_data();
            float* v_data = s.exp_avg_sq.mutable_data();

            for (int64_t i = 0; i < n; i++) {
                float g = grad_data[i];
                if (weight_decay_ > 0.0f) {
                    g += weight_decay_ * param_data[i];
                }
                m_data[i] = beta1_ * m_data[i] + (1.0f - beta1_) * g;
                v_data[i] = beta2_ * v_data[i] + (1.0f - beta2_) * g * g;
                float m_hat = m_data[i] / bc1;
                float v_hat = v_data[i] / bc2;
                param_data[i] -= lr_ * m_hat / (std::sqrt(v_hat) + eps_);
            }
        }
    }
}

// ══════════════════════════════════════════════════════════════════════
// AdamW (decoupled weight decay)
// ══════════════════════════════════════════════════════════════════════

AdamW::AdamW(std::vector<Parameter*> params, float lr,
             std::pair<float, float> betas, float eps,
             float weight_decay, bool amsgrad)
    : lr_(lr),
      beta1_(betas.first),
      beta2_(betas.second),
      eps_(eps),
      weight_decay_(weight_decay),
      amsgrad_(amsgrad) {
    ParamGroup group;
    group.params = std::move(params);
    param_groups_.push_back(std::move(group));
}

void AdamW::step() {
    refresh_params();
    global_step_++;

    float bc1 = 1.0f - std::pow(beta1_, static_cast<float>(global_step_));
    float bc2 = 1.0f - std::pow(beta2_, static_cast<float>(global_step_));

    // GPU-batched path
    bool has_gpu = backend_ && backend_->hasShader("adamw-update");
    if (has_gpu) {
        struct AdamWPush {
            uint32_t total_weights;
            float learning_rate;
            float beta1;
            float beta2;
            float epsilon;
            float weight_decay;
            float beta1_t;      // beta1^t (raw power)
            float beta2_t;      // beta2^t (raw power)
            uint32_t clear_grad; // 1 = zero gradients after update
        };

        float beta1_t = std::pow(beta1_, static_cast<float>(global_step_));
        float beta2_t = std::pow(beta2_, static_cast<float>(global_step_));

        backend_->beginBatch();

        for (auto& group : param_groups_) {
            for (auto* p : group.params) {
                if (!p || !p->has_grad()) { continue; }

                uintptr_t key = reinterpret_cast<uintptr_t>(p);
                auto& s = state_[key];
                int64_t n = p->numel();

                if (s.exp_avg.numel() == 0) {
                    s.exp_avg = Tensor::zeros(p->shape(), backend_);
                    s.exp_avg.ensure_gpu();
                    s.exp_avg_sq = Tensor::zeros(p->shape(), backend_);
                    s.exp_avg_sq.ensure_gpu();
                }
                s.step_count = global_step_;

                uint32_t workgroups = (static_cast<uint32_t>(n) + 255) / 256;

                AdamWPush push{};
                push.total_weights = static_cast<uint32_t>(n);
                push.learning_rate = lr_;
                push.beta1 = beta1_;
                push.beta2 = beta2_;
                push.epsilon = eps_;
                push.weight_decay = weight_decay_;
                push.beta1_t = beta1_t;
                push.beta2_t = beta2_t;
                push.clear_grad = 0;

                if (!p->backend()) p->set_backend(backend_);
                p->ensure_gpu();
                if (!p->grad_ref().backend())
                    p->grad_ref().set_backend(backend_);
                p->grad_ref().ensure_gpu();

                backend_->dispatch(
                    "adamw-update",
                    {p->gpu_handle(), p->grad_ref().gpu_handle(),
                     s.exp_avg.gpu_handle(), s.exp_avg_sq.gpu_handle()},
                    workgroups, 1, 1,
                    &push, sizeof(push));
            }
        }

        backend_->endBatch();
        backend_->barrier();

        for (auto& group : param_groups_) {
            for (auto* p : group.params) {
                if (p && p->has_grad()) {
                    p->mark_gpu_modified();
                }
            }
        }
        return;
    }

    // CPU fallback
    for (auto& group : param_groups_) {
        for (auto* p : group.params) {
            if (!p || !p->has_grad()) { continue; }

            uintptr_t key = reinterpret_cast<uintptr_t>(p);
            auto& s = state_[key];
            int64_t n = p->numel();

            if (s.exp_avg.numel() == 0) {
                s.exp_avg = Tensor::zeros(p->shape(), p->backend());
                s.exp_avg_sq = Tensor::zeros(p->shape(), p->backend());
            }
            s.step_count = global_step_;

            float* param_data = p->mutable_data();
            const float* grad_data = p->grad_ref().data();
            float* m_data = s.exp_avg.mutable_data();
            float* v_data = s.exp_avg_sq.mutable_data();

            for (int64_t i = 0; i < n; i++) {
                if (weight_decay_ > 0.0f) {
                    param_data[i] *= (1.0f - lr_ * weight_decay_);
                }
                float g = grad_data[i];
                m_data[i] = beta1_ * m_data[i] + (1.0f - beta1_) * g;
                v_data[i] = beta2_ * v_data[i] + (1.0f - beta2_) * g * g;
                float m_hat = m_data[i] / bc1;
                float v_hat = v_data[i] / bc2;
                param_data[i] -= lr_ * m_hat / (std::sqrt(v_hat) + eps_);
            }
        }
    }
}

// ══════════════════════════════════════════════════════════════════════
// SGD
// ══════════════════════════════════════════════════════════════════════

SGD::SGD(std::vector<Parameter*> params, float lr, float momentum,
         float weight_decay, bool nesterov)
    : lr_(lr),
      momentum_(momentum),
      weight_decay_(weight_decay),
      nesterov_(nesterov) {
    ParamGroup group;
    group.params = std::move(params);
    param_groups_.push_back(std::move(group));
}

void SGD::step() {
    refresh_params();

    for (auto& group : param_groups_) {
        for (auto* p : group.params) {
            if (!p || !p->has_grad()) { continue; }

            uintptr_t key = reinterpret_cast<uintptr_t>(p);
            auto& s = state_[key];
            int64_t n = p->numel();

            float* param_data = p->mutable_data();
            const float* grad_data = p->grad_ref().data();

            if (momentum_ > 0.0f) {
                if (s.exp_avg.numel() == 0) {
                    s.exp_avg = Tensor::zeros(p->shape(), p->backend());
                }
                float* buf = s.exp_avg.mutable_data();

                for (int64_t i = 0; i < n; i++) {
                    float g = grad_data[i];
                    if (weight_decay_ > 0.0f) {
                        g += weight_decay_ * param_data[i];
                    }
                    buf[i] = momentum_ * buf[i] + g;
                    if (nesterov_) {
                        param_data[i] -= lr_ * (g + momentum_ * buf[i]);
                    } else {
                        param_data[i] -= lr_ * buf[i];
                    }
                }
            } else {
                for (int64_t i = 0; i < n; i++) {
                    float g = grad_data[i];
                    if (weight_decay_ > 0.0f) {
                        g += weight_decay_ * param_data[i];
                    }
                    param_data[i] -= lr_ * g;
                }
            }
        }
    }
}

}  // namespace nn
}  // namespace grilly
