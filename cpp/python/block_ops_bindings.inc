// ═══════════════════════════════════════════════════════════════════════
// CUBEMIND: BLOCK-CODE VSA OPERATIONS
// ═══════════════════════════════════════════════════════════════════════
//
// Include this file inside the PYBIND11_MODULE block in bindings.cpp,
// after the existing cubemind VSA bindings.
//
// Required header (add to the #include section of bindings.cpp):
//   #include "grilly/cubemind/block_ops.h"
//
// Optimizations applied:
//   1. Zero-copy return via py::capsule (no memcpy, no double allocation)
//   2. block_code_bundle accepts stacked 3D array (no Python list iteration)
//   3. GIL released during heavy compute (similarity_batch, codebook, bundle)

    // ── Helper: zero-copy vector → numpy via capsule ──────────────────
    // Transfers ownership of std::vector<float> to Python. No memcpy.

    m.def(
        "block_code_bind",
        [](py::array_t<float> a, py::array_t<float> b) -> py::array_t<float> {
            auto aBuf = a.request();
            auto bBuf = b.request();

            if (aBuf.ndim != 2)
                throw std::runtime_error(
                    "block_code_bind: 'a' must be 2D array (k, l)");
            if (bBuf.ndim != 2)
                throw std::runtime_error(
                    "block_code_bind: 'b' must be 2D array (k, l)");

            uint32_t k = static_cast<uint32_t>(aBuf.shape[0]);
            uint32_t l = static_cast<uint32_t>(aBuf.shape[1]);

            if (static_cast<uint32_t>(bBuf.shape[0]) != k ||
                static_cast<uint32_t>(bBuf.shape[1]) != l)
                throw std::runtime_error(
                    "block_code_bind: shape mismatch between a and b");

            // FIX #1: Zero-copy return via capsule
            auto* result = new std::vector<float>(grilly::cubemind::blockCodeBind(
                static_cast<const float*>(aBuf.ptr),
                static_cast<const float*>(bBuf.ptr), k, l));

            auto capsule = py::capsule(result, [](void* p) {
                delete reinterpret_cast<std::vector<float>*>(p);
            });

            return py::array_t<float>(
                {k, l},
                {l * sizeof(float), sizeof(float)},
                result->data(),
                capsule);
        },
        py::arg("a"), py::arg("b"),
        "Per-block circular convolution binding for sparse block codes");

    m.def(
        "block_code_unbind",
        [](py::array_t<float> composite,
           py::array_t<float> known) -> py::array_t<float> {
            auto cBuf = composite.request();
            auto kBuf = known.request();

            if (cBuf.ndim != 2)
                throw std::runtime_error(
                    "block_code_unbind: 'composite' must be 2D array (k, l)");
            if (kBuf.ndim != 2)
                throw std::runtime_error(
                    "block_code_unbind: 'known' must be 2D array (k, l)");

            uint32_t k = static_cast<uint32_t>(cBuf.shape[0]);
            uint32_t l = static_cast<uint32_t>(cBuf.shape[1]);

            if (static_cast<uint32_t>(kBuf.shape[0]) != k ||
                static_cast<uint32_t>(kBuf.shape[1]) != l)
                throw std::runtime_error(
                    "block_code_unbind: shape mismatch between composite and known");

            // FIX #1: Zero-copy return
            auto* result = new std::vector<float>(grilly::cubemind::blockCodeUnbind(
                static_cast<const float*>(cBuf.ptr),
                static_cast<const float*>(kBuf.ptr), k, l));

            auto capsule = py::capsule(result, [](void* p) {
                delete reinterpret_cast<std::vector<float>*>(p);
            });

            return py::array_t<float>(
                {k, l},
                {l * sizeof(float), sizeof(float)},
                result->data(),
                capsule);
        },
        py::arg("composite"), py::arg("known"),
        "Per-block circular correlation (inverse of bind)");

    // FIX #2: Accept stacked 3D array (n, k, l) instead of Python list.
    // This eliminates 1000x buffer_request() overhead for large bundles.
    m.def(
        "block_code_bundle",
        [](py::array_t<float> vectors,
           bool normalize) -> py::array_t<float> {
            auto vBuf = vectors.request();

            if (vBuf.ndim == 2) {
                // Single vector, just return it
                uint32_t k = static_cast<uint32_t>(vBuf.shape[0]);
                uint32_t l = static_cast<uint32_t>(vBuf.shape[1]);
                auto* result = new std::vector<float>(
                    static_cast<const float*>(vBuf.ptr),
                    static_cast<const float*>(vBuf.ptr) + k * l);
                auto capsule = py::capsule(result, [](void* p) {
                    delete reinterpret_cast<std::vector<float>*>(p);
                });
                return py::array_t<float>(
                    {k, l}, {l * sizeof(float), sizeof(float)},
                    result->data(), capsule);
            }

            if (vBuf.ndim != 3)
                throw std::runtime_error(
                    "block_code_bundle: input must be 3D array (n, k, l) or 2D (k, l)");

            uint32_t n = static_cast<uint32_t>(vBuf.shape[0]);
            uint32_t k = static_cast<uint32_t>(vBuf.shape[1]);
            uint32_t l = static_cast<uint32_t>(vBuf.shape[2]);

            // Build pointer array from contiguous stacked data
            const float* base = static_cast<const float*>(vBuf.ptr);
            std::vector<const float*> ptrs(n);
            for (uint32_t i = 0; i < n; ++i) {
                ptrs[i] = base + i * k * l;
            }

            // FIX #3: Release GIL during compute
            std::vector<float>* result;
            {
                py::gil_scoped_release release;
                result = new std::vector<float>(
                    grilly::cubemind::blockCodeBundle(ptrs, k, l, normalize));
            }

            auto capsule = py::capsule(result, [](void* p) {
                delete reinterpret_cast<std::vector<float>*>(p);
            });

            return py::array_t<float>(
                {k, l}, {l * sizeof(float), sizeof(float)},
                result->data(), capsule);
        },
        py::arg("vectors"), py::arg("normalize") = true,
        "Bundle block-code vectors via element-wise sum with optional L1 "
        "normalization.\nAccepts (n, k, l) stacked array or (k, l) single vector.");

    m.def(
        "block_code_similarity_batch",
        [](py::array_t<float> query,
           py::array_t<float> codebook) -> py::array_t<float> {
            auto qBuf = query.request();
            auto cBuf = codebook.request();

            if (qBuf.ndim != 2)
                throw std::runtime_error(
                    "block_code_similarity_batch: 'query' must be 2D (k, l)");
            if (cBuf.ndim != 3)
                throw std::runtime_error(
                    "block_code_similarity_batch: 'codebook' must be 3D (n, k, l)");

            uint32_t k = static_cast<uint32_t>(qBuf.shape[0]);
            uint32_t l = static_cast<uint32_t>(qBuf.shape[1]);
            uint32_t n = static_cast<uint32_t>(cBuf.shape[0]);

            if (static_cast<uint32_t>(cBuf.shape[1]) != k ||
                static_cast<uint32_t>(cBuf.shape[2]) != l)
                throw std::runtime_error(
                    "block_code_similarity_batch: codebook shape must be (n, " +
                    std::to_string(k) + ", " + std::to_string(l) + ")");

            const float* q_ptr = static_cast<const float*>(qBuf.ptr);
            const float* c_ptr = static_cast<const float*>(cBuf.ptr);

            // FIX #3: Release GIL during batch similarity
            std::vector<float>* result;
            {
                py::gil_scoped_release release;
                result = new std::vector<float>(
                    grilly::cubemind::blockCodeSimilarityBatch(q_ptr, c_ptr, n, k, l));
            }

            auto capsule = py::capsule(result, [](void* p) {
                delete reinterpret_cast<std::vector<float>*>(p);
            });

            return py::array_t<float>(
                {n}, {sizeof(float)}, result->data(), capsule);
        },
        py::arg("query"), py::arg("codebook"),
        "Batch similarity: average per-block dot product between query and codebook");

    m.def(
        "block_code_codebook",
        [](uint32_t k, uint32_t l, uint32_t n,
           uint32_t seed) -> py::array_t<float> {
            // FIX #3: Release GIL during codebook generation
            std::vector<float>* result;
            {
                py::gil_scoped_release release;
                result = new std::vector<float>(
                    grilly::cubemind::blockCodeCodebook(k, l, n, seed));
            }

            auto capsule = py::capsule(result, [](void* p) {
                delete reinterpret_cast<std::vector<float>*>(p);
            });

            return py::array_t<float>(
                {n, k, l},
                {k * l * sizeof(float), l * sizeof(float), sizeof(float)},
                result->data(),
                capsule);
        },
        py::arg("k"), py::arg("l"), py::arg("n"),
        py::arg("seed") = 42,
        "Generate codebook of n discrete block-code vectors via successive binding");

    m.def(
        "block_code_discretize",
        [](py::array_t<float> v) -> py::array_t<float> {
            auto buf = v.request();
            if (buf.ndim != 2)
                throw std::runtime_error(
                    "block_code_discretize: input must be 2D array (k, l)");

            uint32_t k = static_cast<uint32_t>(buf.shape[0]);
            uint32_t l = static_cast<uint32_t>(buf.shape[1]);

            // FIX #1: Zero-copy return
            auto* result = new std::vector<float>(grilly::cubemind::blockCodeDiscretize(
                static_cast<const float*>(buf.ptr), k, l));

            auto capsule = py::capsule(result, [](void* p) {
                delete reinterpret_cast<std::vector<float>*>(p);
            });

            return py::array_t<float>(
                {k, l}, {l * sizeof(float), sizeof(float)},
                result->data(), capsule);
        },
        py::arg("v"),
        "Discretize continuous block-code to one-hot via per-block argmax");

    m.def(
        "block_code_cosine_to_pmf",
        [](py::array_t<float> similarities,
           float temperature) -> py::array_t<float> {
            auto buf = similarities.request();
            if (buf.ndim != 1)
                throw std::runtime_error(
                    "block_code_cosine_to_pmf: input must be 1D array");

            uint32_t n = static_cast<uint32_t>(buf.shape[0]);

            // FIX #1: Zero-copy return
            auto* result = new std::vector<float>(grilly::cubemind::blockCodeCosineTopmf(
                static_cast<const float*>(buf.ptr), n, temperature));

            auto capsule = py::capsule(result, [](void* p) {
                delete reinterpret_cast<std::vector<float>*>(p);
            });

            return py::array_t<float>(
                {n}, {sizeof(float)}, result->data(), capsule);
        },
        py::arg("similarities"), py::arg("temperature") = 40.0f,
        "Convert similarity scores to probability distribution via softmax");
