Performance, Debugging, and Testing
===================================

Performance model
-----------------

Grilly performance depends on:

1. shader availability for your code path
2. memory movement volume (host to device and back)
3. tensor shapes and batch sizing
4. operation fusion opportunities

Design choices
--------------

Performance and correctness tooling in Grilly favors explicitness:

1. Keep kernel boundaries visible so bottlenecks are measurable.
2. Preserve CPU fallback paths for differential testing and debugging.
3. Use strict docs/test builds (`-W` and targeted suites) to catch regressions
   early in CI and local workflows.

Profiling strategy
------------------

Use a layered profiling approach:

1. Measure end-to-end step time.
2. Isolate hotspot operators.
3. Verify whether code path is GPU or fallback CPU.
4. Reduce unnecessary downloads and host-side conversions.

Debugging checklist
-------------------

1. Confirm Vulkan backend initialization.
2. Check tensor dtype (`float32`) and expected shape.
3. Verify required shader exists in loaded shader map.
4. Reproduce issue with smallest possible tensor sizes.
5. Add finite checks (`np.isfinite`) at major boundaries.

Testing workflow
----------------

Useful commands:

.. code-block:: bash

   pytest -q
   pytest tests/experimental -q
   pytest tests/test_integration_vulkan.py -q

For docs:

.. code-block:: bash

   uv run --with-requirements docs/requirements.txt sphinx-build -b html docs docs/_build/html -W

Reproducibility tips
--------------------

- Use stable hash utilities for deterministic seed derivation.
- Save checkpoint artifacts for long ingestion/training flows.
- Keep environment variables and driver versions tracked in experiment logs.
