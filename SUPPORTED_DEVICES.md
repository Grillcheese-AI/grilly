# GRILLY SUPPORTED AND TESTED DEVICES
This list will continue to evolve as we test it on more devices. If you wish to offer your help
to run our tests on specific GPUs, open an Issue and tell us about it. 



AMD

### Desktop (Minimum RDNA2)
RDNA3 and 4 are not listed as they should all be working.

- Radeon RX 6950 XT: Features 16GB of GDDR6 VRAM.
- Radeon RX 6900 XT: Features 16GB of GDDR6 VRAM.
- Radeon RX 6800 XT: Features 16GB of GDDR6 VRAM.
- Radeon RX 6800: Features 16GB of GDDR6 VRAM.
- Radeon RX 6750 XT: Features 12GB of GDDR6 VRAM.
- Radeon RX 6700 XT: Features 12GB of GDDR6 VRAM.
- Radeon RX 6700: This model also has 12GB of GDDR6 VRAM.

**NOTE: AMD RX 6400 and up with 8GB VRAM WILL BE ABLE TO SUPPORT ABOUT 80% OF FEATURES, ONLY MULTIMODAL MIGHT BE A PROBLEM (uses 8-10GB VRAM during tests)**

### Mobile (Minimum RDNA2)
RDNA3 and 4 are not listed as they should all be working.

Mobile GPUs with RDNA 2 architecture and more than 8GB of VRAM include:

- Radeon RX 6850M XT: Has 12GB of VRAM.
- Radeon RX 6800M: Has 12GB of VRAM.
- Radeon RX 6800S: Has 12GB of VRAM.
- Radeon RX 6700M: Has 10GB of VRAM.
- Radeon RX 6700S: Has 10GB of VRAM.


NVIDIA (TESTED):

- A40 (runpod)
- H100 (tensordock)
- RTX 3090 (tensordock)

## HOW TO CHECK IF MY GPU WILL WORK

1. Clone this repository (git clone https://github.com/Grillcheese-AI/grilly)

2. If you do not have UV, install it (pip install uv). On ubuntu, you need to install it with snap (sudo snap install astral-uv --classic)

3. Once uv is installed, make sure you are inside the directory where you cloned grilly then:

```bash
uv venv --python=3.12 # this will make sure you use python 3.12 and not 3.14 which is buggy

# once the virtual environment is created enable it
# on windows: 
.venv\Scripts\activate
# on ubuntu:
source venv/bin/activate

```

4. Build Grilly and install dependencies: 

```bash
# now that your venv is activated enter this command:
uv build .

#this will install the framework in your environment
```

5. Run the following test to know if you can run vulkan:

```bash
uv run pytest/tests/conftest.py
# if you see "Vulkan not available" make sure you installed the vulkan sdk from https://vulkan.lunarg.com/sdk/home
```

6. If you still get not available, and your card is in the minimum supported card, open an issue in this repository and we will look into it. If your card is older than the ones listed here, we will not support them.

7. VEGA and RDNA1 GPU might still work but most of it will run on CPU.
