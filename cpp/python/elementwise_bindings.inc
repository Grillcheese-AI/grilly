    // ═══════════════════════════════════════════════════════════════════════
    // ELEMENT-WISE MATH (replacing numpy for zero-numpy codebase)
    // ═══════════════════════════════════════════════════════════════════════

    m.def(
        "exp",
        [](py::array_t<float> x) -> py::array_t<float> {
            auto buf = x.request();
            py::array_t<float> result(buf.shape);
            auto rbuf = result.request();
            const float* in = static_cast<const float*>(buf.ptr);
            float* out = static_cast<float*>(rbuf.ptr);
            for (py::ssize_t i = 0; i < buf.size; ++i)
                out[i] = std::exp(in[i]);
            return result;
        },
        py::arg("x"), "Element-wise exp");

    m.def(
        "log_safe",
        [](py::array_t<float> x) -> py::array_t<float> {
            auto buf = x.request();
            py::array_t<float> result(buf.shape);
            auto rbuf = result.request();
            const float* in = static_cast<const float*>(buf.ptr);
            float* out = static_cast<float*>(rbuf.ptr);
            for (py::ssize_t i = 0; i < buf.size; ++i)
                out[i] = std::log(std::max(in[i], 1e-30f));
            return result;
        },
        py::arg("x"), "Element-wise log (clamped to avoid -inf)");

    m.def(
        "sqrt_safe",
        [](py::array_t<float> x) -> py::array_t<float> {
            auto buf = x.request();
            py::array_t<float> result(buf.shape);
            auto rbuf = result.request();
            const float* in = static_cast<const float*>(buf.ptr);
            float* out = static_cast<float*>(rbuf.ptr);
            for (py::ssize_t i = 0; i < buf.size; ++i)
                out[i] = std::sqrt(std::max(in[i], 0.0f));
            return result;
        },
        py::arg("x"), "Element-wise sqrt (clamped to non-negative)");

    m.def(
        "clip",
        [](py::array_t<float> x, float lo, float hi) -> py::array_t<float> {
            auto buf = x.request();
            py::array_t<float> result(buf.shape);
            auto rbuf = result.request();
            const float* in = static_cast<const float*>(buf.ptr);
            float* out = static_cast<float*>(rbuf.ptr);
            for (py::ssize_t i = 0; i < buf.size; ++i)
                out[i] = std::clamp(in[i], lo, hi);
            return result;
        },
        py::arg("x"), py::arg("lo"), py::arg("hi"), "Element-wise clip");

    m.def(
        "sign",
        [](py::array_t<float> x) -> py::array_t<float> {
            auto buf = x.request();
            py::array_t<float> result(buf.shape);
            auto rbuf = result.request();
            const float* in = static_cast<const float*>(buf.ptr);
            float* out = static_cast<float*>(rbuf.ptr);
            for (py::ssize_t i = 0; i < buf.size; ++i)
                out[i] = (in[i] > 0.0f) ? 1.0f : ((in[i] < 0.0f) ? -1.0f : 0.0f);
            return result;
        },
        py::arg("x"), "Element-wise sign");

    m.def(
        "abs_val",
        [](py::array_t<float> x) -> py::array_t<float> {
            auto buf = x.request();
            py::array_t<float> result(buf.shape);
            auto rbuf = result.request();
            const float* in = static_cast<const float*>(buf.ptr);
            float* out = static_cast<float*>(rbuf.ptr);
            for (py::ssize_t i = 0; i < buf.size; ++i)
                out[i] = std::abs(in[i]);
            return result;
        },
        py::arg("x"), "Element-wise abs");

    m.def(
        "argmax",
        [](py::array_t<float> x) -> uint32_t {
            auto buf = x.request();
            const float* in = static_cast<const float*>(buf.ptr);
            uint32_t best = 0;
            float best_val = in[0];
            for (py::ssize_t i = 1; i < buf.size; ++i) {
                if (in[i] > best_val) { best_val = in[i]; best = static_cast<uint32_t>(i); }
            }
            return best;
        },
        py::arg("x"), "Argmax over flat array");

    m.def(
        "sum_reduce",
        [](py::array_t<float> x) -> float {
            auto buf = x.request();
            const float* in = static_cast<const float*>(buf.ptr);
            float sum = 0.0f;
            for (py::ssize_t i = 0; i < buf.size; ++i)
                sum += in[i];
            return sum;
        },
        py::arg("x"), "Sum all elements");

    m.def(
        "max_reduce",
        [](py::array_t<float> x) -> float {
            auto buf = x.request();
            const float* in = static_cast<const float*>(buf.ptr);
            float mx = in[0];
            for (py::ssize_t i = 1; i < buf.size; ++i)
                if (in[i] > mx) mx = in[i];
            return mx;
        },
        py::arg("x"), "Max of all elements");

    m.def(
        "zeros",
        [](uint32_t n) -> py::array_t<float> {
            py::array_t<float> result(n);
            auto buf = result.request();
            std::memset(buf.ptr, 0, n * sizeof(float));
            return result;
        },
        py::arg("n"), "Create zero-filled float32 array");

    m.def(
        "ones",
        [](uint32_t n) -> py::array_t<float> {
            py::array_t<float> result(n);
            auto buf = result.request();
            float* out = static_cast<float*>(buf.ptr);
            for (uint32_t i = 0; i < n; ++i) out[i] = 1.0f;
            return result;
        },
        py::arg("n"), "Create ones-filled float32 array");

    m.def(
        "arange",
        [](uint32_t n) -> py::array_t<float> {
            py::array_t<float> result(n);
            auto buf = result.request();
            float* out = static_cast<float*>(buf.ptr);
            for (uint32_t i = 0; i < n; ++i) out[i] = static_cast<float>(i);
            return result;
        },
        py::arg("n"), "Create [0, 1, ..., n-1] float32 array");
