/// bindings_mindforge.cpp — MindForge basis mixer bindings.
///
/// Dispatches three tiny fused shaders used by CubeMind's MindForge
/// hypernetwork to forge LoRA adapters from a shared trainable basis:
///
///   mindforge_basis_mix(coeffs, basis)             → adapter          (fwd)
///   mindforge_bwd_coeff(d_adapter, basis, d_coeffs)                    (bwd)
///   mindforge_bwd_basis(coeffs, d_adapter, d_basis)                    (bwd)
///
/// All inputs/outputs are float32 numpy arrays. The backward ops
/// ACCUMULATE in place (d_coeffs[..] += …, d_basis[..] += …) so the
/// caller must zero those buffers before the first dispatch. This
/// matches the MindForge contract of summing d_A and d_B contributions.

#include "bindings_core.h"

#include <cstring>

namespace {

struct BasisMixPushConstants {
    uint32_t n_basis;
    uint32_t adapter_size;
};

struct BwdCoeffPushConstants {
    uint32_t adapter_size;
};

struct BwdBasisPushConstants {
    uint32_t n_basis;
    uint32_t adapter_size;
};

} // namespace

void register_mindforge_ops(py::module_& m) {

    // ─── Forward: Adapter = Σ coeffs[i] · Basis[i] ───────────────────────
    m.def(
        "mindforge_basis_mix",
        [](GrillyCoreContext& ctx,
           py::array_t<float, py::array::c_style | py::array::forcecast> coeffs,
           py::array_t<float, py::array::c_style | py::array::forcecast> basis)
            -> py::array_t<float> {
            auto cBuf = coeffs.request();
            auto bBuf = basis.request();

            if (cBuf.ndim != 1)
                throw std::runtime_error("coeffs must be 1D (n_basis,)");
            if (bBuf.ndim < 2)
                throw std::runtime_error(
                    "basis must be >= 2D (n_basis, ...)");
            if (bBuf.shape[0] != cBuf.shape[0])
                throw std::runtime_error(
                    "basis.shape[0] must equal coeffs.shape[0]");

            uint32_t n_basis = static_cast<uint32_t>(cBuf.shape[0]);
            if (n_basis > 256)
                throw std::runtime_error(
                    "mindforge_basis_mix: n_basis > 256 not supported "
                    "(shared-memory cache)");

            // Flattened per-basis size (e.g. rank * d_target).
            uint32_t adapter_size = 1;
            for (py::ssize_t i = 1; i < bBuf.ndim; i++)
                adapter_size *= static_cast<uint32_t>(bBuf.shape[i]);

            // Output shape = basis.shape[1:]
            std::vector<py::ssize_t> out_shape(
                bBuf.shape.begin() + 1, bBuf.shape.end());
            py::array_t<float> out_arr(out_shape);
            float* out_ptr = out_arr.mutable_data();

            size_t coeff_bytes   = size_t(n_basis) * sizeof(float);
            size_t basis_bytes   = size_t(n_basis) * adapter_size * sizeof(float);
            size_t adapter_bytes = size_t(adapter_size) * sizeof(float);

            BasisMixPushConstants pc{n_basis, adapter_size};

            {
                py::gil_scoped_release release;
                std::lock_guard<std::mutex> lock(ctx.ctx_mutex);

                auto pe = ctx.cache.getOrCreate(
                    "mindforge-basis-mix", 3, sizeof(pc));

                auto buf_c = ctx.pool.acquire(coeff_bytes);
                auto buf_b = ctx.pool.acquire(basis_bytes);
                auto buf_o = ctx.pool.acquire(adapter_bytes);

                ctx.pool.upload(
                    buf_c, static_cast<const float*>(cBuf.ptr), coeff_bytes);
                ctx.pool.upload(
                    buf_b, static_cast<const float*>(bBuf.ptr), basis_bytes);

                std::vector<VkDescriptorBufferInfo> bufInfos = {
                    {buf_c.handle, 0, (VkDeviceSize)coeff_bytes},
                    {buf_b.handle, 0, (VkDeviceSize)basis_bytes},
                    {buf_o.handle, 0, (VkDeviceSize)adapter_bytes},
                };
                auto dset = ctx.cache.allocDescriptorSet(
                    "mindforge-basis-mix", bufInfos);

                // groupX = ceil(adapter_size / 256)
                uint32_t groups = (adapter_size + 255) / 256;

                ctx.batch.begin();
                ctx.batch.dispatch(pe.pipeline, pe.layout, dset,
                                   groups, 1, 1, &pc, sizeof(pc));
                ctx.batch.submit();

                ctx.pool.download(buf_o, out_ptr, adapter_bytes);

                ctx.pool.release(buf_c);
                ctx.pool.release(buf_b);
                ctx.pool.release(buf_o);
            }

            return out_arr;
        },
        py::arg("device"),
        py::arg("coeffs"),
        py::arg("basis"),
        R"doc(
Forge an adapter by mixing trainable basis matrices.

    adapter = Σ_i coeffs[i] * basis[i]

coeffs: float32 (n_basis,)
basis:  float32 (n_basis, ...) — any trailing shape (e.g. (n_basis, rank, d))
returns float32 with shape == basis.shape[1:]
)doc");

    // ─── Backward: d_coeffs[i] += Σ d_adapter · basis[i] ─────────────────
    m.def(
        "mindforge_bwd_coeff",
        [](GrillyCoreContext& ctx,
           py::array_t<float, py::array::c_style | py::array::forcecast> d_adapter,
           py::array_t<float, py::array::c_style | py::array::forcecast> basis,
           py::array_t<float, py::array::c_style | py::array::forcecast> d_coeffs) {
            auto dBuf = d_adapter.request();
            auto bBuf = basis.request();
            auto cBuf = d_coeffs.request();

            if (cBuf.ndim != 1)
                throw std::runtime_error("d_coeffs must be 1D (n_basis,)");
            if (bBuf.shape[0] != cBuf.shape[0])
                throw std::runtime_error(
                    "basis.shape[0] must equal d_coeffs.shape[0]");

            uint32_t n_basis = static_cast<uint32_t>(cBuf.shape[0]);

            uint32_t adapter_size = 1;
            for (py::ssize_t i = 1; i < bBuf.ndim; i++)
                adapter_size *= static_cast<uint32_t>(bBuf.shape[i]);

            // d_adapter must be the same flat size as basis[0].
            uint32_t d_adapter_flat = 1;
            for (py::ssize_t i = 0; i < dBuf.ndim; i++)
                d_adapter_flat *= static_cast<uint32_t>(dBuf.shape[i]);
            if (d_adapter_flat != adapter_size)
                throw std::runtime_error(
                    "d_adapter flat size must equal product of basis.shape[1:]");

            size_t coeff_bytes   = size_t(n_basis) * sizeof(float);
            size_t basis_bytes   = size_t(n_basis) * adapter_size * sizeof(float);
            size_t adapter_bytes = size_t(adapter_size) * sizeof(float);

            // Writable raw pointer for readback.
            float* dcoeffs_ptr =
                static_cast<float*>(d_coeffs.mutable_unchecked<1>().mutable_data(0));

            BwdCoeffPushConstants pc{adapter_size};

            {
                py::gil_scoped_release release;
                std::lock_guard<std::mutex> lock(ctx.ctx_mutex);

                auto pe = ctx.cache.getOrCreate(
                    "mindforge-bwd-coeff", 3, sizeof(pc));

                auto buf_d = ctx.pool.acquire(adapter_bytes);
                auto buf_b = ctx.pool.acquire(basis_bytes);
                auto buf_c = ctx.pool.acquire(coeff_bytes);

                ctx.pool.upload(
                    buf_d, static_cast<const float*>(dBuf.ptr), adapter_bytes);
                ctx.pool.upload(
                    buf_b, static_cast<const float*>(bBuf.ptr), basis_bytes);
                // Upload the current d_coeffs so the shader's += accumulates
                // onto whatever the caller already had.
                ctx.pool.upload(
                    buf_c, static_cast<const float*>(cBuf.ptr), coeff_bytes);

                std::vector<VkDescriptorBufferInfo> bufInfos = {
                    {buf_d.handle, 0, (VkDeviceSize)adapter_bytes},
                    {buf_b.handle, 0, (VkDeviceSize)basis_bytes},
                    {buf_c.handle, 0, (VkDeviceSize)coeff_bytes},
                };
                auto dset = ctx.cache.allocDescriptorSet(
                    "mindforge-bwd-coeff", bufInfos);

                // One workgroup per basis index.
                ctx.batch.begin();
                ctx.batch.dispatch(pe.pipeline, pe.layout, dset,
                                   n_basis, 1, 1, &pc, sizeof(pc));
                ctx.batch.submit();

                ctx.pool.download(buf_c, dcoeffs_ptr, coeff_bytes);

                ctx.pool.release(buf_d);
                ctx.pool.release(buf_b);
                ctx.pool.release(buf_c);
            }
        },
        py::arg("device"),
        py::arg("d_adapter"),
        py::arg("basis"),
        py::arg("d_coeffs"),
        R"doc(
Accumulate gradient to mixing coefficients: d_coeffs[i] += Σ d_adapter · basis[i]

Writes back into `d_coeffs` in place. Caller must zero it before the
first call if a fresh accumulation is wanted. MindForge calls this
twice (A branch then B branch) to sum both contributions.
)doc");

    // ─── Backward: d_basis[i] += coeffs[i] · d_adapter ───────────────────
    m.def(
        "mindforge_bwd_basis",
        [](GrillyCoreContext& ctx,
           py::array_t<float, py::array::c_style | py::array::forcecast> coeffs,
           py::array_t<float, py::array::c_style | py::array::forcecast> d_adapter,
           py::array_t<float, py::array::c_style | py::array::forcecast> d_basis) {
            auto cBuf = coeffs.request();
            auto dBuf = d_adapter.request();
            auto DBuf = d_basis.request();

            if (cBuf.ndim != 1)
                throw std::runtime_error("coeffs must be 1D (n_basis,)");
            if (DBuf.shape[0] != cBuf.shape[0])
                throw std::runtime_error(
                    "d_basis.shape[0] must equal coeffs.shape[0]");

            uint32_t n_basis = static_cast<uint32_t>(cBuf.shape[0]);
            if (n_basis > 256)
                throw std::runtime_error(
                    "mindforge_bwd_basis: n_basis > 256 not supported");

            uint32_t adapter_size = 1;
            for (py::ssize_t i = 1; i < DBuf.ndim; i++)
                adapter_size *= static_cast<uint32_t>(DBuf.shape[i]);

            uint32_t d_adapter_flat = 1;
            for (py::ssize_t i = 0; i < dBuf.ndim; i++)
                d_adapter_flat *= static_cast<uint32_t>(dBuf.shape[i]);
            if (d_adapter_flat != adapter_size)
                throw std::runtime_error(
                    "d_adapter flat size must equal product of d_basis.shape[1:]");

            size_t coeff_bytes   = size_t(n_basis) * sizeof(float);
            size_t adapter_bytes = size_t(adapter_size) * sizeof(float);
            size_t basis_bytes   = size_t(n_basis) * adapter_size * sizeof(float);

            float* dbasis_ptr = static_cast<float*>(DBuf.ptr);

            BwdBasisPushConstants pc{n_basis, adapter_size};

            {
                py::gil_scoped_release release;
                std::lock_guard<std::mutex> lock(ctx.ctx_mutex);

                auto pe = ctx.cache.getOrCreate(
                    "mindforge-bwd-basis", 3, sizeof(pc));

                auto buf_c = ctx.pool.acquire(coeff_bytes);
                auto buf_d = ctx.pool.acquire(adapter_bytes);
                auto buf_B = ctx.pool.acquire(basis_bytes);

                ctx.pool.upload(
                    buf_c, static_cast<const float*>(cBuf.ptr), coeff_bytes);
                ctx.pool.upload(
                    buf_d, static_cast<const float*>(dBuf.ptr), adapter_bytes);
                // Upload current d_basis for in-place accumulation.
                ctx.pool.upload(
                    buf_B, static_cast<const float*>(DBuf.ptr), basis_bytes);

                std::vector<VkDescriptorBufferInfo> bufInfos = {
                    {buf_c.handle, 0, (VkDeviceSize)coeff_bytes},
                    {buf_d.handle, 0, (VkDeviceSize)adapter_bytes},
                    {buf_B.handle, 0, (VkDeviceSize)basis_bytes},
                };
                auto dset = ctx.cache.allocDescriptorSet(
                    "mindforge-bwd-basis", bufInfos);

                uint32_t groups = (n_basis * adapter_size + 255) / 256;

                ctx.batch.begin();
                ctx.batch.dispatch(pe.pipeline, pe.layout, dset,
                                   groups, 1, 1, &pc, sizeof(pc));
                ctx.batch.submit();

                ctx.pool.download(buf_B, dbasis_ptr, basis_bytes);

                ctx.pool.release(buf_c);
                ctx.pool.release(buf_d);
                ctx.pool.release(buf_B);
            }
        },
        py::arg("device"),
        py::arg("coeffs"),
        py::arg("d_adapter"),
        py::arg("d_basis"),
        R"doc(
Accumulate gradient to basis matrices: d_basis[i] += coeffs[i] * d_adapter

Writes back into `d_basis` in place. Caller must zero it before the
first call if a fresh accumulation is wanted.
)doc");
}
