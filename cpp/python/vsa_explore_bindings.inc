    // ═══════════════════════════════════════════════════════════════════════
    // VSA EXPLORATION + HMM GPU SHADERS
    // ═══════════════════════════════════════════════════════════════════════

    m.def(
        "vsa_explore_transitions",
        [](py::array_t<float> observations,
           py::array_t<float> codebook,
           uint32_t seq_len,
           uint32_t n_trials) -> py::array_t<float> {
            auto obsBuf = observations.request();
            auto cbBuf = codebook.request();

            uint32_t dim = static_cast<uint32_t>(cbBuf.shape[1]);
            uint32_t n_states = static_cast<uint32_t>(cbBuf.shape[0]);

            // CPU fallback: compute transition histogram
            std::vector<float> counts(n_states * n_states, 0.0f);

            const float* obs_ptr = static_cast<const float*>(obsBuf.ptr);
            const float* cb_ptr = static_cast<const float*>(cbBuf.ptr);

            uint32_t k = dim;  // Treat as flat for similarity

            for (uint32_t t = 0; t < seq_len - 1; ++t) {
                // Find closest codebook entry for obs[t]
                float best_a = -1e30f;
                uint32_t state_a = 0;
                for (uint32_t s = 0; s < n_states; ++s) {
                    float sim = 0.0f;
                    for (uint32_t i = 0; i < dim; ++i) {
                        sim += obs_ptr[t * dim + i] * cb_ptr[s * dim + i];
                    }
                    if (sim > best_a) { best_a = sim; state_a = s; }
                }

                // Find closest for obs[t+1]
                float best_b = -1e30f;
                uint32_t state_b = 0;
                for (uint32_t s = 0; s < n_states; ++s) {
                    float sim = 0.0f;
                    for (uint32_t i = 0; i < dim; ++i) {
                        sim += obs_ptr[(t + 1) * dim + i] * cb_ptr[s * dim + i];
                    }
                    if (sim > best_b) { best_b = sim; state_b = s; }
                }

                counts[state_a * n_states + state_b] += 1.0f;
            }

            // Normalize rows
            for (uint32_t i = 0; i < n_states; ++i) {
                float row_sum = 0.0f;
                for (uint32_t j = 0; j < n_states; ++j)
                    row_sum += counts[i * n_states + j];
                if (row_sum > 0.0f) {
                    for (uint32_t j = 0; j < n_states; ++j)
                        counts[i * n_states + j] /= row_sum;
                } else {
                    // Uniform
                    for (uint32_t j = 0; j < n_states; ++j)
                        counts[i * n_states + j] = 1.0f / static_cast<float>(n_states);
                }
            }

            py::array_t<float> result({n_states, n_states});
            std::memcpy(result.request().ptr, counts.data(),
                        n_states * n_states * sizeof(float));
            return result;
        },
        py::arg("observations"), py::arg("codebook"),
        py::arg("seq_len"), py::arg("n_trials") = 1000,
        "VSA exploration: discover transition structure from observation sequences. "
        "Returns normalized transition matrix [n_states, n_states].");
