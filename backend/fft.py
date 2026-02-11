"""
FFT Operations for Vulkan Backend

GPU-accelerated FFT operations:
- FFT (Fast Fourier Transform)
- IFFT (Inverse FFT)
- FFT magnitude
- FFT power spectrum
- FFT normalization

Uses: fft-bitrev.glsl, fft-butterfly.glsl, fft-magnitude.glsl,
      fft-power-spectrum.glsl, fft-normalize.glsl
"""

import struct

import numpy as np

from .base import VULKAN_AVAILABLE, BufferMixin

if VULKAN_AVAILABLE:
    from vulkan import *


class VulkanFFT(BufferMixin):
    """GPU-accelerated FFT operations"""

    def __init__(self, core, pipelines, shaders):
        """Initialize with VulkanCore, VulkanPipelines, and shaders dict"""
        self.core = core
        self.pipelines = pipelines
        self.shaders = shaders

    def fft(self, input_data: np.ndarray, inverse: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute FFT or IFFT.

        Uses: fft-bitrev.glsl, fft-butterfly.glsl

        Args:
            input_data: Input signal (batch, N) - must be power of 2
            inverse: If True, compute IFFT (default: False)

        Returns:
            (real_part, imag_part) - both (batch, N)
        """
        data = input_data.astype(np.float32)

        if data.ndim == 1:
            data = data.reshape(1, -1)

        batch_size, N = data.shape

        # Check if N is power of 2
        if (N & (N - 1)) != 0:
            raise ValueError(f"N must be power of 2, got {N}")

        log2N = int(np.log2(N))

        # Check if shaders are available
        if "fft-bitrev" in self.shaders and "fft-butterfly" in self.shaders:
            # GPU implementation
            data_flat = data.flatten()
            total_elements = batch_size * N

            # Create buffers
            buf_real = self._acquire_buffer(data_flat.nbytes)
            buf_imag = self._acquire_buffer(data_flat.nbytes)

            try:
                # Upload data
                self._upload_buffer(buf_real, data_flat)
                self._upload_buffer(buf_imag, np.zeros_like(data_flat))

                # Pass 1: Bit-reversal permutation
                pipeline_bitrev, layout_bitrev, desc_bitrev = self.pipelines.get_or_create_pipeline(
                    "fft-bitrev", 3, push_constant_size=8
                )
                desc_set_bitrev = self.pipelines._create_descriptor_set(
                    desc_bitrev,
                    [
                        (self._get_buffer_handle(buf_real), data_flat.nbytes),
                        (self._get_buffer_handle(buf_imag), data_flat.nbytes),
                        (self._get_buffer_handle(buf_real), data_flat.nbytes),
                    ],
                )
                push_constants = struct.pack("II", batch_size, N)
                workgroups = (total_elements + 255) // 256
                self.core._dispatch_compute(
                    pipeline_bitrev, layout_bitrev, desc_set_bitrev, workgroups, push_constants
                )

                # Pass 2: Butterfly operations for each stage
                pipeline_butterfly, layout_butterfly, desc_butterfly = (
                    self.pipelines.get_or_create_pipeline("fft-butterfly", 2, push_constant_size=20)
                )
                for stage in range(log2N):
                    desc_set_butterfly = self.pipelines._create_descriptor_set(
                        desc_butterfly,
                        [
                            (self._get_buffer_handle(buf_real), data_flat.nbytes),
                            (self._get_buffer_handle(buf_imag), data_flat.nbytes),
                        ],
                    )
                    push_constants = struct.pack("IIII", batch_size, N, stage, 1 if inverse else 0)
                    workgroups = (batch_size * N // 2 + 255) // 256
                    self.core._dispatch_compute(
                        pipeline_butterfly,
                        layout_butterfly,
                        desc_set_butterfly,
                        workgroups,
                        push_constants,
                    )
                    vkFreeDescriptorSets(
                        self.core.device, self.core.descriptor_pool, 1, [desc_set_butterfly]
                    )

                # Download results
                real_part = self._download_buffer(buf_real, data_flat.nbytes, np.float32)
                imag_part = self._download_buffer(buf_imag, data_flat.nbytes, np.float32)
                real_part = real_part[:total_elements].reshape(batch_size, N)
                imag_part = imag_part[:total_elements].reshape(batch_size, N)

                # Cleanup descriptor set
                vkFreeDescriptorSets(
                    self.core.device, self.core.descriptor_pool, 1, [desc_set_bitrev]
                )

                return real_part, imag_part
            finally:
                self._release_buffers([buf_real, buf_imag])
        else:
            # CPU fallback
            if inverse:
                result = np.fft.ifft(data, axis=-1)
            else:
                result = np.fft.fft(data, axis=-1)

            real_part = result.real.astype(np.float32)
            imag_part = result.imag.astype(np.float32)

            return real_part, imag_part

    def fft_magnitude(self, real_part: np.ndarray, imag_part: np.ndarray) -> np.ndarray:
        """
        Compute FFT magnitude.

        Uses: fft-magnitude.glsl

        Args:
            real_part: Real part of FFT (batch, N)
            imag_part: Imaginary part of FFT (batch, N)

        Returns:
            Magnitude (batch, N)
        """
        if "fft-magnitude" in self.shaders:
            # GPU implementation
            real_flat = real_part.astype(np.float32).flatten()
            imag_flat = imag_part.astype(np.float32).flatten()
            total_elements = len(real_flat)

            # Create buffers
            buf_real = self._acquire_buffer(real_flat.nbytes)
            buf_imag = self._acquire_buffer(imag_flat.nbytes)
            buf_mag = self._acquire_buffer(real_flat.nbytes)

            try:
                # Upload
                self._upload_buffer(buf_real, real_flat)
                self._upload_buffer(buf_imag, imag_flat)

                # Get pipeline
                pipeline, layout, desc_layout = self.pipelines.get_or_create_pipeline(
                    "fft-magnitude", 4, push_constant_size=8
                )
                desc_set = self.pipelines._create_descriptor_set(
                    desc_layout,
                    [
                        (self._get_buffer_handle(buf_real), real_flat.nbytes),
                        (self._get_buffer_handle(buf_imag), imag_flat.nbytes),
                        (self._get_buffer_handle(buf_mag), real_flat.nbytes),
                        (self._get_buffer_handle(buf_mag), real_flat.nbytes),
                    ],  # Phase buffer same as mag for now
                )

                # Dispatch
                push_constants = struct.pack("II", total_elements, 0)  # compute_phase = 0
                workgroups = (total_elements + 255) // 256
                self.core._dispatch_compute(pipeline, layout, desc_set, workgroups, push_constants)

                # Download
                magnitude = self._download_buffer(buf_mag, real_flat.nbytes, np.float32)
                magnitude = magnitude[:total_elements].reshape(real_part.shape)

                # Cleanup descriptor set
                vkFreeDescriptorSets(self.core.device, self.core.descriptor_pool, 1, [desc_set])

                return magnitude
            finally:
                self._release_buffers([buf_real, buf_imag, buf_mag])
        else:
            # CPU fallback
            return np.sqrt(real_part**2 + imag_part**2).astype(np.float32)

    def fft_power_spectrum(self, real_part: np.ndarray, imag_part: np.ndarray) -> np.ndarray:
        """
        Compute FFT power spectrum.

        Uses: fft-power-spectrum.glsl

        Args:
            real_part: Real part of FFT (batch, N)
            imag_part: Imaginary part of FFT (batch, N)

        Returns:
            Power spectrum (batch, N)
        """
        if "fft-power-spectrum" in self.shaders:
            # GPU implementation
            real_flat = real_part.astype(np.float32).flatten()
            imag_flat = imag_part.astype(np.float32).flatten()
            total_elements = len(real_flat)
            N = real_part.shape[-1]

            # Create buffers
            buf_real = self._acquire_buffer(real_flat.nbytes)
            buf_imag = self._acquire_buffer(imag_flat.nbytes)
            buf_power = self._acquire_buffer(real_flat.nbytes)

            try:
                # Upload
                self._upload_buffer(buf_real, real_flat)
                self._upload_buffer(buf_imag, imag_flat)

                # Get pipeline
                pipeline, layout, desc_layout = self.pipelines.get_or_create_pipeline(
                    "fft-power-spectrum", 3, push_constant_size=12
                )
                desc_set = self.pipelines._create_descriptor_set(
                    desc_layout,
                    [
                        (self._get_buffer_handle(buf_real), real_flat.nbytes),
                        (self._get_buffer_handle(buf_imag), imag_flat.nbytes),
                        (self._get_buffer_handle(buf_power), real_flat.nbytes),
                    ],
                )

                # Dispatch
                push_constants = struct.pack("III", total_elements, 0, N)  # scale_by_n = 0
                workgroups = (total_elements + 255) // 256
                self.core._dispatch_compute(pipeline, layout, desc_set, workgroups, push_constants)

                # Download
                power = self._download_buffer(buf_power, real_flat.nbytes, np.float32)
                power = power[:total_elements].reshape(real_part.shape)

                # Cleanup descriptor set
                vkFreeDescriptorSets(self.core.device, self.core.descriptor_pool, 1, [desc_set])

                return power
            finally:
                self._release_buffers([buf_real, buf_imag, buf_power])
        else:
            # CPU fallback
            magnitude = self.fft_magnitude(real_part, imag_part)
            return (magnitude**2).astype(np.float32)

    def fft_normalize(
        self, real_part: np.ndarray, imag_part: np.ndarray, N: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Normalize FFT output.

        Uses: fft-normalize.glsl

        Args:
            real_part: Real part of FFT (batch, N)
            imag_part: Imaginary part of FFT (batch, N)
            N: Signal length

        Returns:
            (normalized_real, normalized_imag)
        """
        if "fft-normalize" in self.shaders:
            # GPU implementation
            real_flat = real_part.astype(np.float32).flatten()
            imag_flat = imag_part.astype(np.float32).flatten()
            total_elements = len(real_flat)

            # Create buffers (in-place)
            buf_real = self._acquire_buffer(real_flat.nbytes)
            buf_imag = self._acquire_buffer(imag_flat.nbytes)

            try:
                # Upload
                self._upload_buffer(buf_real, real_flat)
                self._upload_buffer(buf_imag, imag_flat)

                # Get pipeline
                pipeline, layout, desc_layout = self.pipelines.get_or_create_pipeline(
                    "fft-normalize", 2, push_constant_size=8
                )
                desc_set = self.pipelines._create_descriptor_set(
                    desc_layout,
                    [
                        (self._get_buffer_handle(buf_real), real_flat.nbytes),
                        (self._get_buffer_handle(buf_imag), imag_flat.nbytes),
                    ],
                )

                # Dispatch
                push_constants = struct.pack("II", total_elements, N)
                workgroups = (total_elements + 255) // 256
                self.core._dispatch_compute(pipeline, layout, desc_set, workgroups, push_constants)

                # Download
                normalized_real = self._download_buffer(buf_real, real_flat.nbytes, np.float32)
                normalized_imag = self._download_buffer(buf_imag, imag_flat.nbytes, np.float32)
                normalized_real = normalized_real[:total_elements].reshape(real_part.shape)
                normalized_imag = normalized_imag[:total_elements].reshape(imag_part.shape)

                # Cleanup descriptor set
                vkFreeDescriptorSets(self.core.device, self.core.descriptor_pool, 1, [desc_set])

                return normalized_real, normalized_imag
            finally:
                self._release_buffers([buf_real, buf_imag])
        else:
            # CPU fallback
            scale = 1.0 / np.sqrt(N)
            return (real_part * scale).astype(np.float32), (imag_part * scale).astype(np.float32)
