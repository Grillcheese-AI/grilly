#include "grilly/nn/optimizer.h"

#include <cmath>
#include <cstring>
#include <stdexcept>

namespace grilly {
namespace nn {

// ══════════════════════════════════════════════════════════════════════
// Optimizer base
// ══════════════════════════════════════════════════════════════════════

void Optimizer::zero_grad() {
    for (auto& group : param_groups_) {
        for (auto* p : group.params) {
            p->zero_grad();
        }
    }
}

size_t Optimizer::param_count() const {
    size_t count = 0;
    for (const auto& group : param_groups_) {
        count += group.params.size();
    }
    return count;
}

py::dict Optimizer::state_dict() const {
    py::dict state;
    py::list param_states;

    for (const auto& [param, pstate] : state_) {
        py::dict ps;
        ps["step_count"] = pstate.step_count;
        if (pstate.exp_avg.valid()) {
            ps["exp_avg"] = pstate.exp_avg.numpy();
        }
        if (pstate.exp_avg_sq.valid()) {
            ps["exp_avg_sq"] = pstate.exp_avg_sq.numpy();
        }
        param_states.append(ps);
    }

    state["param_states"] = param_states;
    return state;
}

void Optimizer::load_state_dict(py::dict state) {
    // Simplified — full implementation would match params by index
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

    // Detect backend from first parameter
    if (!param_groups_.empty() && !param_groups_[0].params.empty()) {
        backend_ = param_groups_[0].params[0]->backend();
    }
}

void Adam::step() {
    global_step_++;

    // Bias correction factors
    float bc1 = 1.0f - std::pow(beta1_, static_cast<float>(global_step_));
    float bc2 = 1.0f - std::pow(beta2_, static_cast<float>(global_step_));

    // Try GPU-batched path
    if (backend_ && backend_->hasShader("adam-update")) {
        struct AdamPush {
            uint32_t num_elements;
            float lr;
            float beta1;
            float beta2;
            float eps;
            float weight_decay;
            float bc1;
            float bc2;
        };

        backend_->beginBatch();

        for (auto& group : param_groups_) {
            for (auto* p : group.params) {
                if (!p->has_grad()) continue;

                auto& s = state_[p];
                int64_t n = p->numel();

                // Initialize state tensors if needed
                if (!s.exp_avg.valid()) {
                    s.exp_avg = Tensor::zeros(p->shape(), backend_);
                    s.exp_avg.ensure_gpu();
                    s.exp_avg_sq = Tensor::zeros(p->shape(), backend_);
                    s.exp_avg_sq.ensure_gpu();
                }
                s.step_count = global_step_;

                uint32_t workgroups = (static_cast<uint32_t>(n) + 255) / 256;

                AdamPush push{};
                push.num_elements = static_cast<uint32_t>(n);
                push.lr = lr_;
                push.beta1 = beta1_;
                push.beta2 = beta2_;
                push.eps = eps_;
                push.weight_decay = weight_decay_;
                push.bc1 = bc1;
                push.bc2 = bc2;

                // Ensure all tensors are on GPU
                p->ensure_gpu();
                p->grad_ref().ensure_gpu();

                backend_->dispatch(
                    "adam-update",
                    {p->gpu_handle(), p->grad_ref().gpu_handle(),
                     s.exp_avg.gpu_handle(), s.exp_avg_sq.gpu_handle()},
                    workgroups, 1, 1,
                    &push, sizeof(push));
                backend_->barrier();
            }
        }

        backend_->endBatch();  // ★ ONE fence wait for ALL parameters

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
            if (!p->has_grad()) continue;

            auto& s = state_[p];
            int64_t n = p->numel();

            // Initialize state tensors if needed
            if (!s.exp_avg.valid()) {
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

                // L2 regularization (weight decay for Adam, not AdamW)
                if (weight_decay_ > 0.0f) {
                    g += weight_decay_ * param_data[i];
                }

                // Update biased first moment estimate
                m_data[i] = beta1_ * m_data[i] + (1.0f - beta1_) * g;

                // Update biased second raw moment estimate
                v_data[i] = beta2_ * v_data[i] + (1.0f - beta2_) * g * g;

                // Bias-corrected estimates
                float m_hat = m_data[i] / bc1;
                float v_hat = v_data[i] / bc2;

                // Update parameters
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

    if (!param_groups_.empty() && !param_groups_[0].params.empty()) {
        backend_ = param_groups_[0].params[0]->backend();
    }
}

void AdamW::step() {
    global_step_++;

    float bc1 = 1.0f - std::pow(beta1_, static_cast<float>(global_step_));
    float bc2 = 1.0f - std::pow(beta2_, static_cast<float>(global_step_));

    for (auto& group : param_groups_) {
        for (auto* p : group.params) {
            if (!p->has_grad()) continue;

            auto& s = state_[p];
            int64_t n = p->numel();

            if (!s.exp_avg.valid()) {
                s.exp_avg = Tensor::zeros(p->shape(), p->backend());
                s.exp_avg_sq = Tensor::zeros(p->shape(), p->backend());
            }
            s.step_count = global_step_;

            float* param_data = p->mutable_data();
            const float* grad_data = p->grad_ref().data();
            float* m_data = s.exp_avg.mutable_data();
            float* v_data = s.exp_avg_sq.mutable_data();

            for (int64_t i = 0; i < n; i++) {
                // Decoupled weight decay (applied to params, not gradients)
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

    if (!param_groups_.empty() && !param_groups_[0].params.empty()) {
        backend_ = param_groups_[0].params[0]->backend();
    }
}

void SGD::step() {
    for (auto& group : param_groups_) {
        for (auto* p : group.params) {
            if (!p->has_grad()) continue;

            auto& s = state_[p];
            int64_t n = p->numel();

            float* param_data = p->mutable_data();
            const float* grad_data = p->grad_ref().data();

            if (momentum_ > 0.0f) {
                // Initialize momentum buffer if needed
                if (!s.exp_avg.valid()) {
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
