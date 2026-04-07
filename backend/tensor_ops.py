"""
Core tensor ops for Vulkan backend: transpose, cat, where, index_select, bmm.

GPU dispatch path uses the SPIR-V shaders compiled from:
  shaders/tensor-transpose.glsl
  shaders/tensor-cat.glsl
  shaders/tensor-where.glsl
  shaders/tensor-index-select.glsl
  shaders/tensor-bmm.glsl

Falls back to numpy when Vulkan is unavailable or the shader is missing.
"""

import struct

import numpy as np

from .base import BufferMixin, VULKAN_PYTHON_BINDINGS_AVAILABLE

if VULKAN_PYTHON_BINDINGS_AVAILABLE:
    from vulkan import *  # noqa: F401,F403


class VulkanTensorOps(BufferMixin):
    """Fundamental tensor manipulation ops: transpose, cat, where, index_select, bmm."""

    def __init__(self, core, pipelines, shaders):
        self.core = core
        self.pipelines = pipelines
        self.shaders = shaders

    # ── transpose ────────────────────────────────────────────────────────

    def transpose(self, x):
        """Transpose a 2D matrix.

        Args:
            x: numpy array of shape (rows, cols).

        Returns:
            numpy float32 array of shape (cols, rows).
        """
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2:
            return np.ascontiguousarray(x.T, dtype=np.float32)
        if "tensor-transpose" not in self.shaders:
            return np.ascontiguousarray(x.T, dtype=np.float32)

        rows, cols = x.shape
        in_bytes = rows * cols * 4
        out_bytes = cols * rows * 4

        buf_in, release_in = self._prepare_input(x, size=in_bytes)
        buf_out = self._acquire_buffer(out_bytes)

        try:
            pipeline, layout, _ = self.pipelines.get_or_create_pipeline(
                "tensor-transpose", 2, push_constant_size=8
            )
            in_handle = self._get_buffer_handle(buf_in)
            out_handle = self._get_buffer_handle(buf_out)
            desc = self.pipelines.get_cached_descriptor_set(
                "tensor-transpose",
                [(in_handle, in_bytes), (out_handle, out_bytes)],
            )
            push = struct.pack("2I", rows, cols)
            workgroups = (rows * cols + 255) // 256
            self.core._dispatch_compute(pipeline, layout, desc, workgroups, push)
            result = self._download_buffer(buf_out, out_bytes, np.float32)
            return result.reshape(cols, rows)
        finally:
            if release_in:
                self._release_buffer(buf_in)
            self._release_buffer(buf_out)

    # ── cat ──────────────────────────────────────────────────────────────

    def cat(self, a, b, axis=-1):
        """Concatenate two tensors along their last axis.

        Only the last-axis case is GPU-accelerated; all others fall back to
        numpy.

        Args:
            a:    numpy array.
            b:    numpy array with the same shape except on the concat axis.
            axis: concatenation axis (default -1 = last).

        Returns:
            numpy float32 array.
        """
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        ndim = a.ndim
        if axis < 0:
            axis = ndim + axis
        if axis != ndim - 1 or "tensor-cat" not in self.shaders:
            return np.concatenate([a, b], axis=axis).astype(np.float32)

        batch_size = int(np.prod(a.shape[:-1]))
        dim_a = a.shape[-1]
        dim_b = b.shape[-1]
        total_out = batch_size * (dim_a + dim_b)

        in_a_bytes = batch_size * dim_a * 4
        in_b_bytes = batch_size * dim_b * 4
        out_bytes = total_out * 4

        buf_a, release_a = self._prepare_input(a, size=in_a_bytes)
        buf_b, release_b = self._prepare_input(b, size=in_b_bytes)
        buf_out = self._acquire_buffer(out_bytes)

        try:
            pipeline, layout, _ = self.pipelines.get_or_create_pipeline(
                "tensor-cat", 3, push_constant_size=12
            )
            a_handle = self._get_buffer_handle(buf_a)
            b_handle = self._get_buffer_handle(buf_b)
            out_handle = self._get_buffer_handle(buf_out)
            desc = self.pipelines.get_cached_descriptor_set(
                "tensor-cat",
                [(a_handle, in_a_bytes), (b_handle, in_b_bytes), (out_handle, out_bytes)],
            )
            push = struct.pack("3I", batch_size, dim_a, dim_b)
            workgroups = (total_out + 255) // 256
            self.core._dispatch_compute(pipeline, layout, desc, workgroups, push)
            result = self._download_buffer(buf_out, out_bytes, np.float32)
            out_shape = a.shape[:-1] + (dim_a + dim_b,)
            return result.reshape(out_shape)
        finally:
            if release_a:
                self._release_buffer(buf_a)
            if release_b:
                self._release_buffer(buf_b)
            self._release_buffer(buf_out)

    # ── where ────────────────────────────────────────────────────────────

    def where(self, cond, a, b):
        """Elementwise conditional: out[i] = cond[i] > 0 ? a[i] : b[i].

        Args:
            cond: float32 condition array.
            a:    float32 array for true branch.
            b:    float32 array for false branch.

        Returns:
            numpy float32 array of same shape.
        """
        cond = np.asarray(cond, dtype=np.float32)
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        if "tensor-where" not in self.shaders:
            return np.where(cond > 0, a, b).astype(np.float32)

        count = int(cond.size)
        nbytes = count * 4

        buf_cond, release_cond = self._prepare_input(cond, size=nbytes)
        buf_a, release_a = self._prepare_input(a, size=nbytes)
        buf_b, release_b = self._prepare_input(b, size=nbytes)
        buf_out = self._acquire_buffer(nbytes)

        try:
            pipeline, layout, _ = self.pipelines.get_or_create_pipeline(
                "tensor-where", 4, push_constant_size=4
            )
            cond_handle = self._get_buffer_handle(buf_cond)
            a_handle = self._get_buffer_handle(buf_a)
            b_handle = self._get_buffer_handle(buf_b)
            out_handle = self._get_buffer_handle(buf_out)
            desc = self.pipelines.get_cached_descriptor_set(
                "tensor-where",
                [
                    (cond_handle, nbytes),
                    (a_handle, nbytes),
                    (b_handle, nbytes),
                    (out_handle, nbytes),
                ],
            )
            push = struct.pack("I", count)
            workgroups = (count + 255) // 256
            self.core._dispatch_compute(pipeline, layout, desc, workgroups, push)
            result = self._download_buffer(buf_out, nbytes, np.float32)
            return result.reshape(cond.shape)
        finally:
            if release_cond:
                self._release_buffer(buf_cond)
            if release_a:
                self._release_buffer(buf_a)
            if release_b:
                self._release_buffer(buf_b)
            self._release_buffer(buf_out)

    # ── index_select ─────────────────────────────────────────────────────

    def index_select(self, x, indices):
        """Gather rows by integer indices: out[i] = x[indices[i]].

        Args:
            x:       2-D float32 array of shape (num_rows, row_size).
            indices: 1-D integer array.

        Returns:
            numpy float32 array of shape (len(indices), row_size).
        """
        x = np.asarray(x, dtype=np.float32)
        indices = np.asarray(indices, dtype=np.uint32)
        if not indices.flags["C_CONTIGUOUS"]:
            indices = np.ascontiguousarray(indices)
        if "tensor-index-select" not in self.shaders:
            return x[indices]

        num_indices = int(indices.size)
        row_size = x.shape[-1] if x.ndim > 1 else 1

        in_bytes = int(x.size) * 4
        idx_bytes = num_indices * 4  # uint32
        out_bytes = num_indices * row_size * 4

        buf_in, release_in = self._prepare_input(x, size=in_bytes)
        buf_idx, release_idx = self._prepare_input(indices, size=idx_bytes)
        buf_out = self._acquire_buffer(out_bytes)

        try:
            pipeline, layout, _ = self.pipelines.get_or_create_pipeline(
                "tensor-index-select", 3, push_constant_size=8
            )
            in_handle = self._get_buffer_handle(buf_in)
            idx_handle = self._get_buffer_handle(buf_idx)
            out_handle = self._get_buffer_handle(buf_out)
            desc = self.pipelines.get_cached_descriptor_set(
                "tensor-index-select",
                [(in_handle, in_bytes), (idx_handle, idx_bytes), (out_handle, out_bytes)],
            )
            push = struct.pack("2I", num_indices, row_size)
            workgroups = (num_indices * row_size + 255) // 256
            self.core._dispatch_compute(pipeline, layout, desc, workgroups, push)
            result = self._download_buffer(buf_out, out_bytes, np.float32)
            return result.reshape(num_indices, row_size)
        finally:
            if release_in:
                self._release_buffer(buf_in)
            if release_idx:
                self._release_buffer(buf_idx)
            self._release_buffer(buf_out)

    # ── bmm ──────────────────────────────────────────────────────────────

    def bmm(self, a, b):
        """Batched matrix multiply: out[i] = a[i] @ b[i].

        Args:
            a: float32 array of shape (batch, M, K).
            b: float32 array of shape (batch, K, N).

        Returns:
            numpy float32 array of shape (batch, M, N).
        """
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        if "tensor-bmm" not in self.shaders:
            return np.einsum("bik,bkj->bij", a, b).astype(np.float32)

        batch, M, K = a.shape
        K2, N = b.shape[1], b.shape[2]
        assert K == K2, f"tensor_bmm: K mismatch {K} vs {K2}"

        a_bytes = batch * M * K * 4
        b_bytes = batch * K * N * 4
        out_bytes = batch * M * N * 4

        buf_a, release_a = self._prepare_input(a, size=a_bytes)
        buf_b, release_b = self._prepare_input(b, size=b_bytes)
        buf_out = self._acquire_buffer(out_bytes)

        try:
            pipeline, layout, _ = self.pipelines.get_or_create_pipeline(
                "tensor-bmm", 3, push_constant_size=16
            )
            a_handle = self._get_buffer_handle(buf_a)
            b_handle = self._get_buffer_handle(buf_b)
            out_handle = self._get_buffer_handle(buf_out)
            desc = self.pipelines.get_cached_descriptor_set(
                "tensor-bmm",
                [(a_handle, a_bytes), (b_handle, b_bytes), (out_handle, out_bytes)],
            )
            push = struct.pack("4I", batch, M, K, N)
            # Dispatch (ceil(M/16), ceil(N/16), batch)
            gx = (M + 15) // 16
            gy = (N + 15) // 16
            gz = batch
            self.core._dispatch_compute(pipeline, layout, desc, gx, push, gy, gz)
            result = self._download_buffer(buf_out, out_bytes, np.float32)
            return result.reshape(batch, M, N)
        finally:
            if release_a:
                self._release_buffer(buf_a)
            if release_b:
                self._release_buffer(buf_b)
            self._release_buffer(buf_out)
