    // ═══════════════════════════════════════════════════════════════════════
    // CUBEMIND: TENSOR OPS (replacing numpy for zero-numpy codebase)
    // Implementation: cpp/src/cubemind/tensor_ops.cpp (Eigen SIMD + BLAS)
    // ═══════════════════════════════════════════════════════════════════════

    // ── Allocation ───────────────────────────────────────────────

    m.def("t_zeros", [](int64_t n) {
        auto v = grilly::cubemind::zeros(n);
        py::array_t<float> r(n);
        std::memcpy(r.mutable_data(), v.data(), n * sizeof(float));
        return r;
    }, py::arg("n"), "Zero-filled float32 array");

    m.def("t_ones", [](int64_t n) {
        auto v = grilly::cubemind::ones(n);
        py::array_t<float> r(n);
        std::memcpy(r.mutable_data(), v.data(), n * sizeof(float));
        return r;
    }, py::arg("n"), "Ones-filled float32 array");

    m.def("t_full", [](int64_t n, float val) {
        auto v = grilly::cubemind::full(n, val);
        py::array_t<float> r(n);
        std::memcpy(r.mutable_data(), v.data(), n * sizeof(float));
        return r;
    }, py::arg("n"), py::arg("val"), "Filled float32 array");

    m.def("t_arange", [](int64_t n) {
        auto v = grilly::cubemind::arange(n);
        py::array_t<float> r(n);
        std::memcpy(r.mutable_data(), v.data(), n * sizeof(float));
        return r;
    }, py::arg("n"), "[0..n-1] float32");

    // ── Element-wise Math (Eigen SIMD) ───────────────────────────

    m.def("t_exp", [](py::array_t<float> x) {
        auto buf = x.request();
        int64_t n = buf.size;
        auto v = grilly::cubemind::exp(static_cast<const float*>(buf.ptr), n);
        py::array_t<float> r(buf.shape);
        std::memcpy(r.mutable_data(), v.data(), n * sizeof(float));
        return r;
    }, py::arg("x"), "Element-wise exp");

    m.def("t_log", [](py::array_t<float> x) {
        auto buf = x.request();
        int64_t n = buf.size;
        auto v = grilly::cubemind::log_safe(static_cast<const float*>(buf.ptr), n);
        py::array_t<float> r(buf.shape);
        std::memcpy(r.mutable_data(), v.data(), n * sizeof(float));
        return r;
    }, py::arg("x"), "Element-wise log (safe)");

    m.def("t_sqrt", [](py::array_t<float> x) {
        auto buf = x.request();
        int64_t n = buf.size;
        auto v = grilly::cubemind::sqrt_safe(static_cast<const float*>(buf.ptr), n);
        py::array_t<float> r(buf.shape);
        std::memcpy(r.mutable_data(), v.data(), n * sizeof(float));
        return r;
    }, py::arg("x"), "Element-wise sqrt (safe)");

    m.def("t_abs", [](py::array_t<float> x) {
        auto buf = x.request();
        int64_t n = buf.size;
        auto v = grilly::cubemind::abs(static_cast<const float*>(buf.ptr), n);
        py::array_t<float> r(buf.shape);
        std::memcpy(r.mutable_data(), v.data(), n * sizeof(float));
        return r;
    }, py::arg("x"), "Element-wise abs");

    m.def("t_sign", [](py::array_t<float> x) {
        auto buf = x.request();
        int64_t n = buf.size;
        auto v = grilly::cubemind::sign(static_cast<const float*>(buf.ptr), n);
        py::array_t<float> r(buf.shape);
        std::memcpy(r.mutable_data(), v.data(), n * sizeof(float));
        return r;
    }, py::arg("x"), "Element-wise sign");

    m.def("t_clip", [](py::array_t<float> x, float lo, float hi) {
        auto buf = x.request();
        int64_t n = buf.size;
        auto v = grilly::cubemind::clip(static_cast<const float*>(buf.ptr), n, lo, hi);
        py::array_t<float> r(buf.shape);
        std::memcpy(r.mutable_data(), v.data(), n * sizeof(float));
        return r;
    }, py::arg("x"), py::arg("lo"), py::arg("hi"), "Element-wise clip");

    m.def("t_pow", [](py::array_t<float> x, float p) {
        auto buf = x.request();
        int64_t n = buf.size;
        auto v = grilly::cubemind::pow(static_cast<const float*>(buf.ptr), n, p);
        py::array_t<float> r(buf.shape);
        std::memcpy(r.mutable_data(), v.data(), n * sizeof(float));
        return r;
    }, py::arg("x"), py::arg("p"), "Element-wise power");

    // ── Reductions (Eigen SIMD) ──────────────────────────────────

    m.def("t_sum", [](py::array_t<float> x) {
        auto buf = x.request();
        return grilly::cubemind::sum(static_cast<const float*>(buf.ptr), buf.size);
    }, py::arg("x"), "Sum all elements");

    m.def("t_mean", [](py::array_t<float> x) {
        auto buf = x.request();
        return grilly::cubemind::mean(static_cast<const float*>(buf.ptr), buf.size);
    }, py::arg("x"), "Mean");

    m.def("t_var", [](py::array_t<float> x) {
        auto buf = x.request();
        return grilly::cubemind::var(static_cast<const float*>(buf.ptr), buf.size);
    }, py::arg("x"), "Variance");

    m.def("t_max", [](py::array_t<float> x) {
        auto buf = x.request();
        return grilly::cubemind::max(static_cast<const float*>(buf.ptr), buf.size);
    }, py::arg("x"), "Max element");

    m.def("t_min", [](py::array_t<float> x) {
        auto buf = x.request();
        return grilly::cubemind::min(static_cast<const float*>(buf.ptr), buf.size);
    }, py::arg("x"), "Min element");

    m.def("t_argmax", [](py::array_t<float> x) {
        auto buf = x.request();
        return grilly::cubemind::argmax(static_cast<const float*>(buf.ptr), buf.size);
    }, py::arg("x"), "Index of max");

    m.def("t_argmin", [](py::array_t<float> x) {
        auto buf = x.request();
        return grilly::cubemind::argmin(static_cast<const float*>(buf.ptr), buf.size);
    }, py::arg("x"), "Index of min");

    m.def("t_argsort", [](py::array_t<float> x) {
        auto buf = x.request();
        auto v = grilly::cubemind::argsort(static_cast<const float*>(buf.ptr), buf.size);
        py::array_t<int64_t> r(static_cast<int64_t>(v.size()));
        std::memcpy(r.mutable_data(), v.data(), v.size() * sizeof(int64_t));
        return r;
    }, py::arg("x"), "Indices that would sort");

    // ── Linear Algebra (Eigen BLAS) ──────────────────────────────

    m.def("t_dot", [](py::array_t<float> a, py::array_t<float> b) {
        auto ab = a.request(); auto bb = b.request();
        return grilly::cubemind::dot(
            static_cast<const float*>(ab.ptr),
            static_cast<const float*>(bb.ptr), ab.size);
    }, py::arg("a"), py::arg("b"), "Dot product");

    m.def("t_norm_l2", [](py::array_t<float> x) {
        auto buf = x.request();
        return grilly::cubemind::norm_l2(static_cast<const float*>(buf.ptr), buf.size);
    }, py::arg("x"), "L2 norm");

    m.def("t_norm_l1", [](py::array_t<float> x) {
        auto buf = x.request();
        return grilly::cubemind::norm_l1(static_cast<const float*>(buf.ptr), buf.size);
    }, py::arg("x"), "L1 norm");

    m.def("t_matmul", [](py::array_t<float> a, py::array_t<float> b) {
        auto ab = a.request(); auto bb = b.request();
        if (ab.ndim != 2 || bb.ndim != 2)
            throw std::runtime_error("t_matmul: both inputs must be 2D");
        int64_t m = ab.shape[0], k = ab.shape[1], n = bb.shape[1];
        auto v = grilly::cubemind::matmul(
            static_cast<const float*>(ab.ptr),
            static_cast<const float*>(bb.ptr), m, k, n);
        py::array_t<float> r({m, n});
        std::memcpy(r.mutable_data(), v.data(), m * n * sizeof(float));
        return r;
    }, py::arg("a"), py::arg("b"), "Matrix multiply (Eigen BLAS)");

    // ── FFT ──────────────────────────────────────────────────────

    m.def("t_fft_1d", [](py::array_t<float> x) {
        auto buf = x.request();
        int64_t n = buf.size;
        py::array_t<float> re(n), im(n);
        grilly::cubemind::fft_1d(
            static_cast<const float*>(buf.ptr),
            static_cast<float*>(re.mutable_data()),
            static_cast<float*>(im.mutable_data()), n);
        return py::make_tuple(re, im);
    }, py::arg("x"), "1D DFT -> (real, imag)");

    m.def("t_ifft_1d", [](py::array_t<float> re, py::array_t<float> im) {
        auto rb = re.request();
        int64_t n = rb.size;
        py::array_t<float> r(n);
        grilly::cubemind::ifft_1d(
            static_cast<const float*>(rb.ptr),
            static_cast<const float*>(im.request().ptr),
            static_cast<float*>(r.mutable_data()), n);
        return r;
    }, py::arg("real"), py::arg("imag"), "Inverse 1D DFT (real out)");

    // ── RNG ──────────────────────────────────────────────────────

    m.def("t_rand_uniform", [](int64_t n, uint32_t seed) {
        auto v = grilly::cubemind::rand_uniform(n, seed);
        py::array_t<float> r(n);
        std::memcpy(r.mutable_data(), v.data(), n * sizeof(float));
        return r;
    }, py::arg("n"), py::arg("seed") = 42, "Uniform [0,1) random");

    m.def("t_rand_normal", [](int64_t n, uint32_t seed) {
        auto v = grilly::cubemind::rand_normal(n, seed);
        py::array_t<float> r(n);
        std::memcpy(r.mutable_data(), v.data(), n * sizeof(float));
        return r;
    }, py::arg("n"), py::arg("seed") = 42, "Standard normal random");

    m.def("t_rand_int", [](int64_t n, int64_t low, int64_t high, uint32_t seed) {
        auto v = grilly::cubemind::rand_int(n, low, high, seed);
        py::array_t<int64_t> r(n);
        std::memcpy(r.mutable_data(), v.data(), n * sizeof(int64_t));
        return r;
    }, py::arg("n"), py::arg("low"), py::arg("high"), py::arg("seed") = 42,
       "Random ints in [low, high)");
