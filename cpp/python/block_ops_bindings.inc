// ═══════════════════════════════════════════════════════════════════════
// CUBEMIND: BLOCK-CODE VSA OPERATIONS
// ═══════════════════════════════════════════════════════════════════════
//
// Include this file inside the PYBIND11_MODULE block in bindings.cpp,
// after the existing cubemind VSA bindings.
//
// Required header (add to the #include section of bindings.cpp):
//   #include "grilly/cubemind/block_ops.h"

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

            auto result = grilly::cubemind::blockCodeBind(
                static_cast<const float*>(aBuf.ptr),
                static_cast<const float*>(bBuf.ptr), k, l);

            py::array_t<float> arr({k, l});
            std::memcpy(arr.request().ptr, result.data(),
                        static_cast<size_t>(k) * l * sizeof(float));
            return arr;
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

            auto result = grilly::cubemind::blockCodeUnbind(
                static_cast<const float*>(cBuf.ptr),
                static_cast<const float*>(kBuf.ptr), k, l);

            py::array_t<float> arr({k, l});
            std::memcpy(arr.request().ptr, result.data(),
                        static_cast<size_t>(k) * l * sizeof(float));
            return arr;
        },
        py::arg("composite"), py::arg("known"),
        "Per-block circular correlation (inverse of bind)");

    m.def(
        "block_code_bundle",
        [](std::vector<py::array_t<float>> vectors,
           bool normalize) -> py::array_t<float> {
            if (vectors.empty())
                throw std::runtime_error(
                    "block_code_bundle: empty vector list");

            auto firstBuf = vectors[0].request();
            if (firstBuf.ndim != 2)
                throw std::runtime_error(
                    "block_code_bundle: vectors must be 2D arrays (k, l)");

            uint32_t k = static_cast<uint32_t>(firstBuf.shape[0]);
            uint32_t l = static_cast<uint32_t>(firstBuf.shape[1]);

            std::vector<const float*> ptrs;
            ptrs.reserve(vectors.size());
            for (size_t i = 0; i < vectors.size(); ++i) {
                auto buf = vectors[i].request();
                if (buf.ndim != 2 ||
                    static_cast<uint32_t>(buf.shape[0]) != k ||
                    static_cast<uint32_t>(buf.shape[1]) != l)
                    throw std::runtime_error(
                        "block_code_bundle: all vectors must have shape (" +
                        std::to_string(k) + ", " + std::to_string(l) + ")");
                ptrs.push_back(static_cast<const float*>(buf.ptr));
            }

            auto result = grilly::cubemind::blockCodeBundle(ptrs, k, l, normalize);

            py::array_t<float> arr({k, l});
            std::memcpy(arr.request().ptr, result.data(),
                        static_cast<size_t>(k) * l * sizeof(float));
            return arr;
        },
        py::arg("vectors"), py::arg("normalize") = true,
        "Bundle block-code vectors via element-wise sum with optional L1 normalization");

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

            auto result = grilly::cubemind::blockCodeSimilarityBatch(
                static_cast<const float*>(qBuf.ptr),
                static_cast<const float*>(cBuf.ptr), n, k, l);

            py::array_t<float> arr(n);
            std::memcpy(arr.request().ptr, result.data(),
                        n * sizeof(float));
            return arr;
        },
        py::arg("query"), py::arg("codebook"),
        "Batch similarity: average per-block dot product between query and codebook");

    m.def(
        "block_code_codebook",
        [](uint32_t k, uint32_t l, uint32_t n,
           uint32_t seed) -> py::array_t<float> {
            auto result = grilly::cubemind::blockCodeCodebook(k, l, n, seed);

            py::array_t<float> arr({n, k, l});
            std::memcpy(arr.request().ptr, result.data(),
                        static_cast<size_t>(n) * k * l * sizeof(float));
            return arr;
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

            auto result = grilly::cubemind::blockCodeDiscretize(
                static_cast<const float*>(buf.ptr), k, l);

            py::array_t<float> arr({k, l});
            std::memcpy(arr.request().ptr, result.data(),
                        static_cast<size_t>(k) * l * sizeof(float));
            return arr;
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

            auto result = grilly::cubemind::blockCodeCosineTopmf(
                static_cast<const float*>(buf.ptr), n, temperature);

            py::array_t<float> arr(n);
            std::memcpy(arr.request().ptr, result.data(),
                        n * sizeof(float));
            return arr;
        },
        py::arg("similarities"), py::arg("temperature") = 40.0f,
        "Convert similarity scores to probability distribution via softmax");
