Installation
============

Step 1: Install GPU drivers
---------------------------

Install a Vulkan-capable driver for your GPU (AMD, NVIDIA, or Intel).
This is required for GPU execution.

Step 2: Create a Python environment
-----------------------------------

Use a clean environment:

.. code-block:: bash

   python -m venv .venv
   .venv\Scripts\activate

Step 3: Install Grilly
----------------------

Install from PyPI:

.. code-block:: bash

   pip install grilly

Or install from source:

.. code-block:: bash

   git clone https://github.com/grillcheese-ai/grilly.git
   cd grilly
   pip install -e .

Step 4: Verify your setup
-------------------------

Run a minimal check:

.. code-block:: python

   import grilly

   print("Vulkan available:", grilly.VULKAN_AVAILABLE)
   backend = grilly.Compute()
   print("Backend initialized:", type(backend).__name__)

Step 5: Optional environment flags
----------------------------------

Useful runtime flags:

.. code-block:: bash

   # Select GPU index on multi-GPU hosts
   set VK_GPU_INDEX=0

   # Allow CPU Vulkan adapters if no discrete GPU is available
   set ALLOW_CPU_VULKAN=1
