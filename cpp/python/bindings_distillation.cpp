/// bindings_distillation.cpp — Fused KD+CE distillation loss binding.
///
/// Dispatches distillation-loss.glsl: computes both loss and gradients in a
/// single GPU dispatch per token using subgroup arithmetic reductions.

#include "bindings_core.h"

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
           float alpha) -> std::pair<py::array_t<float>, py::array_t<float>> {
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
                throw std::runtime_error("teacher_logits shape must match student_logits");
            if (static_cast<uint32_t>(lBuf.shape[0]) != seq_len)
                throw std::runtime_error("labels length must match seq_len");

            uint32_t logit_bytes = seq_len * vocab_size * sizeof(float);
            uint32_t label_bytes = seq_len * sizeof(int32_t);
            uint32_t grad_bytes = logit_bytes;
            uint32_t loss_bytes = seq_len * sizeof(float);

            // Allocate output arrays
            py::array_t<float> grad_out({(ssize_t)seq_len, (ssize_t)vocab_size});
            py::array_t<float> loss_out((ssize_t)seq_len);
            auto gBuf = grad_out.request();
            auto loBuf = loss_out.request();

            DistillationPushConstants pc{vocab_size, temperature, alpha};

            {
                py::gil_scoped_release release;
                std::lock_guard<std::mutex> lock(ctx.ctx_mutex);

                // Get or create pipeline
                auto [pipeline, layout, dsetLayout] =
                    ctx.cache.getOrCreatePipeline("distillation-loss", 5, sizeof(pc));

                // Acquire GPU buffers
                auto buf_s = ctx.pool.acquire(logit_bytes);
                auto buf_t = ctx.pool.acquire(logit_bytes);
                auto buf_l = ctx.pool.acquire(label_bytes);
                auto buf_g = ctx.pool.acquire(grad_bytes);
                auto buf_lo = ctx.pool.acquire(loss_bytes);

                // Upload inputs
                ctx.pool.upload(buf_s, static_cast<const float*>(sBuf.ptr), logit_bytes);
                ctx.pool.upload(buf_t, static_cast<const float*>(tBuf.ptr), logit_bytes);
                ctx.pool.upload(buf_l, static_cast<const int32_t*>(lBuf.ptr), label_bytes);

                // Descriptor set
                auto dset = ctx.cache.allocateDescriptorSet(dsetLayout);
                VkDescriptorBufferInfo infos[5] = {
                    {ctx.pool.vkBuffer(buf_s), 0, logit_bytes},
                    {ctx.pool.vkBuffer(buf_t), 0, logit_bytes},
                    {ctx.pool.vkBuffer(buf_l), 0, label_bytes},
                    {ctx.pool.vkBuffer(buf_g), 0, grad_bytes},
                    {ctx.pool.vkBuffer(buf_lo), 0, loss_bytes},
                };
                VkWriteDescriptorSet writes[5];
                for (int i = 0; i < 5; i++) {
                    writes[i] = {};
                    writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    writes[i].dstSet = dset;
                    writes[i].dstBinding = i;
                    writes[i].descriptorCount = 1;
                    writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                    writes[i].pBufferInfo = &infos[i];
                }
                vkUpdateDescriptorSets(ctx.device.logicalDevice(), 5, writes, 0, nullptr);

                // Record and dispatch: 1 workgroup per token
                ctx.batch.begin();
                ctx.batch.bindPipeline(pipeline);
                ctx.batch.bindDescriptorSet(layout, dset);
                ctx.batch.pushConstants(layout, VK_SHADER_STAGE_COMPUTE_BIT,
                                         0, sizeof(pc), &pc);
                ctx.batch.dispatch(seq_len, 1, 1);
                ctx.batch.submit();
                ctx.batch.waitFence();

                // Download results
                ctx.pool.download(buf_g, static_cast<float*>(gBuf.ptr), grad_bytes);
                ctx.pool.download(buf_lo, static_cast<float*>(loBuf.ptr), loss_bytes);

                // Release buffers
                ctx.pool.release(buf_s);
                ctx.pool.release(buf_t);
                ctx.pool.release(buf_l);
                ctx.pool.release(buf_g);
                ctx.pool.release(buf_lo);
            }

            return {loss_out, grad_out};
        },
        py::arg("device"),
        py::arg("student_logits"),
        py::arg("teacher_logits"),
        py::arg("labels"),
        py::arg("temperature") = 2.0f,
        py::arg("alpha") = 0.6f,
        R"doc(
Fused Knowledge Distillation + Cross-Entropy loss on GPU.

Computes in a single dispatch per token:
  loss = (1-alpha)*CE + alpha*T^2*KL
  grad = (1-alpha)*grad_CE + alpha*grad_KD

Args:
    device: GrillyCoreContext (Device)
    student_logits: (seq_len, vocab) float32
    teacher_logits: (seq_len, vocab) float32
    labels: (seq_len,) int32
    temperature: softmax temperature for KD (default 2.0)
    alpha: KD weight, loss = (1-alpha)*CE + alpha*KD (default 0.6)

Returns:
    (loss, grad) where loss is (seq_len,) and grad is (seq_len, vocab)
)doc");
}
