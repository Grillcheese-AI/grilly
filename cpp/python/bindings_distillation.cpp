/// bindings_distillation.cpp — Fused KD+CE distillation loss binding.
///
/// Dispatches distillation-loss.glsl: computes both loss and gradients in a
/// single GPU dispatch per token using subgroup arithmetic reductions.

#include "bindings_core.h"

#include <cstring>

struct DistillationPushConstants {
    uint32_t vocab_size;
    float temperature;
    float alpha;
};

void register_distillation_ops(py::module_& m) {

    m.def(
        "distillation_loss",
        [](GrillyCoreContext& ctx,
           py::array_t<float> student_logits,
           py::array_t<float> teacher_logits,
           py::array_t<int32_t> labels,
           float temperature,
           float alpha) -> py::dict {
            auto sBuf = student_logits.request();
            auto tBuf = teacher_logits.request();
            auto lBuf = labels.request();
            require_c_contiguous_float(sBuf);
            require_c_contiguous_float(tBuf);

            if (sBuf.ndim != 2)
                throw std::runtime_error("student_logits must be 2D (seq_len, vocab)");
            if (tBuf.ndim != 2)
                throw std::runtime_error("teacher_logits must be 2D (seq_len, vocab)");

            uint32_t seq_len = static_cast<uint32_t>(sBuf.shape[0]);
            uint32_t vocab_size = static_cast<uint32_t>(sBuf.shape[1]);

            if (static_cast<uint32_t>(tBuf.shape[0]) != seq_len ||
                static_cast<uint32_t>(tBuf.shape[1]) != vocab_size)
                throw std::runtime_error("teacher shape must match student");

            size_t logit_bytes = size_t(seq_len) * vocab_size * sizeof(float);
            size_t label_bytes = size_t(seq_len) * sizeof(int32_t);
            size_t loss_bytes  = size_t(seq_len) * sizeof(float);

            // Output arrays
            py::array_t<float> grad_out({(ssize_t)seq_len, (ssize_t)vocab_size});
            py::array_t<float> loss_out((ssize_t)seq_len);

            DistillationPushConstants pc{vocab_size, temperature, alpha};

            {
                py::gil_scoped_release release;
                std::lock_guard<std::mutex> lock(ctx.ctx_mutex);

                // Pipeline
                auto pe = ctx.cache.getOrCreate("distillation-loss", 5, sizeof(pc));

                // Buffers
                auto buf_s  = ctx.pool.acquire(logit_bytes);
                auto buf_t  = ctx.pool.acquire(logit_bytes);
                auto buf_l  = ctx.pool.acquire(label_bytes);
                auto buf_g  = ctx.pool.acquire(logit_bytes);
                auto buf_lo = ctx.pool.acquire(loss_bytes);

                // Upload student + teacher logits
                ctx.pool.upload(buf_s, static_cast<const float*>(sBuf.ptr), logit_bytes);
                ctx.pool.upload(buf_t, static_cast<const float*>(tBuf.ptr), logit_bytes);
                // Labels: raw memcpy (int32, not float)
                std::memcpy(buf_l.mappedPtr, lBuf.ptr, label_bytes);

                // Descriptor set
                std::vector<VkDescriptorBufferInfo> bufInfos = {
                    {buf_s.handle,  0, (VkDeviceSize)logit_bytes},
                    {buf_t.handle,  0, (VkDeviceSize)logit_bytes},
                    {buf_l.handle,  0, (VkDeviceSize)label_bytes},
                    {buf_g.handle,  0, (VkDeviceSize)logit_bytes},
                    {buf_lo.handle, 0, (VkDeviceSize)loss_bytes},
                };
                auto dset = ctx.cache.allocDescriptorSet("distillation-loss", bufInfos);

                // Dispatch: 1 workgroup (1024 threads) per token
                ctx.batch.begin();
                ctx.batch.dispatch(pe.pipeline, pe.layout, dset,
                                   seq_len, 1, 1, &pc, sizeof(pc));
                ctx.batch.submit();

                // Download results
                ctx.pool.download(buf_g,  grad_out.mutable_data(), logit_bytes);
                ctx.pool.download(buf_lo, loss_out.mutable_data(), loss_bytes);

                // Release
                ctx.pool.release(buf_s);
                ctx.pool.release(buf_t);
                ctx.pool.release(buf_l);
                ctx.pool.release(buf_g);
                ctx.pool.release(buf_lo);
            }

            py::dict result;
            result["loss"] = loss_out;
            result["grad"] = grad_out;
            return result;
        },
        py::arg("device"),
        py::arg("student_logits"),
        py::arg("teacher_logits"),
        py::arg("labels"),
        py::arg("temperature") = 2.0f,
        py::arg("alpha") = 0.6f,
        R"doc(
Fused KD+CE distillation loss on GPU. Returns dict with 'loss' and 'grad'.

loss = (1-alpha)*CE + alpha*T^2*KL per token
grad = (1-alpha)*grad_CE + alpha*grad_KD per (token, vocab)
)doc");
}
