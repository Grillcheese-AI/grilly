    // ═══════════════════════════════════════════════════════════════════════
    // CUBEMIND: HMM FORWARD-BACKWARD AND BAUM-WELCH
    // ═══════════════════════════════════════════════════════════════════════

    m.def(
        "hmm_forward",
        [](py::array_t<float> log_A, py::array_t<float> log_pi,
           py::array_t<float> log_B) -> py::tuple {
            auto aBuf = log_A.request();
            auto piBuf = log_pi.request();
            auto bBuf = log_B.request();

            if (aBuf.ndim != 2)
                throw std::runtime_error("hmm_forward: log_A must be 2D (n, n)");
            if (piBuf.ndim != 1)
                throw std::runtime_error("hmm_forward: log_pi must be 1D (n,)");
            if (bBuf.ndim != 2)
                throw std::runtime_error("hmm_forward: log_B must be 2D (T, n)");

            uint32_t n = static_cast<uint32_t>(aBuf.shape[0]);
            uint32_t T = static_cast<uint32_t>(bBuf.shape[0]);

            auto result = grilly::cubemind::hmmForward(
                static_cast<const float*>(aBuf.ptr),
                static_cast<const float*>(piBuf.ptr),
                static_cast<const float*>(bBuf.ptr), T, n);

            py::array_t<float> alpha({T, n});
            std::memcpy(alpha.request().ptr, result.log_alpha.data(),
                        T * n * sizeof(float));

            return py::make_tuple(alpha, result.log_likelihood);
        },
        py::arg("log_A"), py::arg("log_pi"), py::arg("log_B"),
        "HMM forward algorithm in log-space. Returns (log_alpha, log_likelihood).");

    m.def(
        "hmm_forward_backward",
        [](py::array_t<float> log_A, py::array_t<float> log_pi,
           py::array_t<float> log_B) -> py::tuple {
            auto aBuf = log_A.request();
            auto piBuf = log_pi.request();
            auto bBuf = log_B.request();

            uint32_t n = static_cast<uint32_t>(aBuf.shape[0]);
            uint32_t T = static_cast<uint32_t>(bBuf.shape[0]);

            auto result = grilly::cubemind::hmmForwardBackward(
                static_cast<const float*>(aBuf.ptr),
                static_cast<const float*>(piBuf.ptr),
                static_cast<const float*>(bBuf.ptr), T, n);

            py::array_t<float> alpha({T, n});
            py::array_t<float> beta({T, n});
            py::array_t<float> gamma({T, n});
            std::memcpy(alpha.request().ptr, result.log_alpha.data(), T * n * sizeof(float));
            std::memcpy(beta.request().ptr, result.log_beta.data(), T * n * sizeof(float));
            std::memcpy(gamma.request().ptr, result.log_gamma.data(), T * n * sizeof(float));

            return py::make_tuple(alpha, beta, gamma, result.log_likelihood);
        },
        py::arg("log_A"), py::arg("log_pi"), py::arg("log_B"),
        "HMM forward-backward. Returns (log_alpha, log_beta, log_gamma, log_likelihood).");

    m.def(
        "hmm_baum_welch_batch",
        [](py::array_t<float> log_A, py::array_t<float> log_pi,
           py::array_t<float> batch_log_B,
           py::array_t<uint32_t> seq_lengths,
           float smoothing) -> py::tuple {
            auto aBuf = log_A.request();
            auto piBuf = log_pi.request();
            auto bBuf = batch_log_B.request();
            auto lenBuf = seq_lengths.request();

            uint32_t n = static_cast<uint32_t>(aBuf.shape[0]);
            uint32_t n_seqs = static_cast<uint32_t>(lenBuf.shape[0]);

            auto result = grilly::cubemind::hmmBaumWelchBatch(
                static_cast<const float*>(aBuf.ptr),
                static_cast<const float*>(piBuf.ptr),
                static_cast<const float*>(bBuf.ptr),
                static_cast<const uint32_t*>(lenBuf.ptr),
                n_seqs, n, smoothing);

            py::array_t<float> A_num({n, n});
            py::array_t<float> pi_num(n);
            std::memcpy(A_num.request().ptr, result.A_num.data(), n * n * sizeof(float));
            std::memcpy(pi_num.request().ptr, result.pi_num.data(), n * sizeof(float));

            return py::make_tuple(A_num, pi_num, result.avg_log_likelihood);
        },
        py::arg("log_A"), py::arg("log_pi"), py::arg("batch_log_B"),
        py::arg("seq_lengths"), py::arg("smoothing") = 1e-3f,
        "Baum-Welch E-step + M-step accumulation over a batch. "
        "Returns (A_num, pi_num, avg_log_likelihood).");
