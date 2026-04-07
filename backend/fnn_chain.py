"""
Batched FNN recording: multiple linear/relu/softmax dispatches, one submit + one fence wait.

Use ``VulkanCompute.record_commands(fnn_chain=True)`` or ``VulkanFNN.chain_record()``.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Union

import numpy as np

from .core import CommandRecorder

if TYPE_CHECKING:
    from .fnn import VulkanFNN


@dataclass(frozen=True)
class ChainBufferHandle:
    """GPU buffer produced inside a chain; pass to the next ``rec.*`` call or ``rec.read()``."""

    buf: Any
    nbytes: int
    shape: tuple[int, ...]

    @property
    def ndim(self) -> int:
        return len(self.shape)


class FnnChainRecorder(CommandRecorder):
    """Records linear / ReLU / softmax into the batch command buffer without per-op submit.

    Buffers acquired for numpy inputs and intermediate outputs are released when
    ``read()`` runs or when the context exits.
    """

    def __init__(self, fnn: VulkanFNN):
        super().__init__(fnn.core)
        self._fnn = fnn
        self._owned: list[Any] = []
        self._submitted = False
        self._released = False

    def _track(self, buf: Any) -> None:
        if buf is not None:
            self._owned.append(buf)

    def _release_owned(self) -> None:
        if self._released:
            return
        for buf in self._owned:
            try:
                self._fnn._release_buffer(buf)
            except Exception:
                pass
        self._owned.clear()
        self._released = True

    def linear(
        self,
        x: Union[np.ndarray, ChainBufferHandle, Any],
        weights: np.ndarray,
        bias: np.ndarray | None = None,
    ) -> ChainBufferHandle:
        """Record ``fnn-linear`` dispatch. Returns a handle to the output buffer (no download)."""
        if not self._recording:
            raise RuntimeError("linear() must be called inside a chain_record context")
        fnn = self._fnn
        if "fnn-linear" not in fnn.shaders:
            raise RuntimeError("fnn-linear shader not available")

        from ..utils.tensor_conversion import VulkanTensor

        w_np = np.ascontiguousarray(weights, dtype=np.float32)
        output_dim, input_dim = w_np.shape

        if isinstance(x, ChainBufferHandle):
            buf_input = x.buf
            original_shape = x.shape
            if len(original_shape) > 2:
                batch_seq = int(np.prod(original_shape[:-1]))
            else:
                batch_seq = original_shape[0]
            if original_shape[-1] != input_dim:
                raise ValueError(
                    f"ChainBufferHandle last dim {original_shape[-1]} != weight input_dim {input_dim}"
                )
            input_nbytes = batch_seq * input_dim * 4
        elif isinstance(x, VulkanTensor):
            original_shape = x.shape
            if len(original_shape) > 2:
                batch_seq = int(np.prod(original_shape[:-1]))
            else:
                batch_seq = original_shape[0]
            input_nbytes = batch_seq * input_dim * 4
            buf_input, release_input = fnn._prepare_input(x, size=input_nbytes)
            if release_input:
                self._track(buf_input)
        else:
            x_np = np.asarray(x, dtype=np.float32)
            original_shape = x_np.shape
            if len(original_shape) > 2:
                batch_seq = int(np.prod(original_shape[:-1]))
                x_2d = x_np.reshape(batch_seq, input_dim)
            else:
                batch_seq = original_shape[0]
                x_2d = x_np
            x_flat = np.ascontiguousarray(x_2d).ravel()
            input_nbytes = x_flat.nbytes
            buf_input = fnn._acquire_buffer(input_nbytes)
            fnn._upload_buffer(buf_input, x_flat)
            self._track(buf_input)
            release_input = True

        w_nbytes = int(w_np.size) * 4
        output_size = batch_seq * output_dim * 4

        buf_weights, _ = fnn._get_or_upload_weight(w_np)
        has_bias = 1 if bias is not None else 0
        if bias is not None:
            bias_np = np.ascontiguousarray(bias, dtype=np.float32)
            buf_bias, _ = fnn._get_or_upload_weight(bias_np)
            bias_flat = bias_np.ravel()
            bias_nbytes = bias_flat.nbytes
        else:
            buf_bias = fnn._acquire_buffer(4)
            fnn._upload_buffer(buf_bias, np.zeros(1, dtype=np.float32))
            self._track(buf_bias)
            bias_flat = None
            bias_nbytes = 4

        buf_output = fnn._acquire_buffer(output_size)
        self._track(buf_output)

        pipeline, pipeline_layout, _ = fnn.pipelines.get_or_create_pipeline(
            "fnn-linear", 4, push_constant_size=16
        )
        descriptor_set = fnn.pipelines.get_cached_descriptor_set(
            "fnn-linear",
            [
                (fnn._get_buffer_handle(buf_input), input_nbytes),
                (fnn._get_buffer_handle(buf_weights), w_nbytes),
                (
                    fnn._get_buffer_handle(buf_bias),
                    bias_nbytes,
                ),
                (fnn._get_buffer_handle(buf_output), output_size),
            ],
        )
        push = struct.pack("IIII", batch_seq, input_dim, output_dim, has_bias)
        gx = (output_dim + 15) // 16
        gy = (batch_seq + 15) // 16
        self.dispatch(pipeline, pipeline_layout, descriptor_set, (gx, gy), push)
        self.barrier()

        if len(original_shape) > 2:
            out_shape = original_shape[:-1] + (output_dim,)
        else:
            out_shape = (batch_seq, output_dim)

        return ChainBufferHandle(buf_output, output_size, out_shape)

    def relu(self, h: ChainBufferHandle) -> ChainBufferHandle:
        """Record ``activation-relu`` dispatch."""
        if not self._recording:
            raise RuntimeError("relu() must be called inside a chain_record context")
        fnn = self._fnn
        if "activation-relu" not in fnn.shaders:
            raise RuntimeError("activation-relu shader not available")

        total_elements = h.nbytes // 4
        buf_out = fnn._acquire_buffer(h.nbytes)
        self._track(buf_out)

        pipeline, pipeline_layout, _ = fnn.pipelines.get_or_create_pipeline(
            "activation-relu", 2, push_constant_size=4
        )
        descriptor_set = fnn.pipelines.get_cached_descriptor_set(
            "activation-relu",
            [
                (fnn._get_buffer_handle(h.buf), h.nbytes),
                (fnn._get_buffer_handle(buf_out), h.nbytes),
            ],
        )
        push = struct.pack("I", total_elements)
        wg = (total_elements + 255) // 256
        self.dispatch(pipeline, pipeline_layout, descriptor_set, (wg,), push)
        self.barrier()

        return ChainBufferHandle(buf_out, h.nbytes, h.shape)

    def softmax(self, h: ChainBufferHandle, dim: int = -1) -> ChainBufferHandle:
        """Record ``activation-softmax`` (3 passes). ``dim`` must be the last axis (-1)."""
        if not self._recording:
            raise RuntimeError("softmax() must be called inside a chain_record context")
        if dim not in (-1, len(h.shape) - 1):
            raise NotImplementedError(
                "FnnChainRecorder.softmax only supports softmax over the last dimension (dim=-1)"
            )
        fnn = self._fnn
        if "activation-softmax" not in fnn.shaders:
            raise RuntimeError("activation-softmax shader not available")

        shape = h.shape
        if len(shape) == 1:
            batch_size, seq_len, features = 1, 1, shape[0]
        elif len(shape) == 2:
            batch_size, seq_len, features = shape[0], 1, shape[1]
        else:
            batch_size, seq_len, features = shape[0], shape[1], int(np.prod(shape[2:]))

        data_nbytes = h.nbytes
        buf_max = fnn._acquire_buffer(batch_size * seq_len * 4)
        buf_sum = fnn._acquire_buffer(batch_size * seq_len * 4)
        self._track(buf_max)
        self._track(buf_sum)

        buf_out = fnn._acquire_buffer(data_nbytes)
        self._track(buf_out)

        pipeline, pipeline_layout, _ = fnn.pipelines.get_or_create_pipeline(
            "activation-softmax", 4, push_constant_size=24
        )
        descriptor_set = fnn.pipelines.get_cached_descriptor_set(
            "activation-softmax",
            [
                (fnn._get_buffer_handle(h.buf), data_nbytes),
                (fnn._get_buffer_handle(buf_out), data_nbytes),
                (fnn._get_buffer_handle(buf_max), batch_size * seq_len * 4),
                (fnn._get_buffer_handle(buf_sum), batch_size * seq_len * 4),
            ],
        )

        workgroups_bs = ((batch_size * seq_len) + 255) // 256
        workgroups_flat = (int(np.prod(shape)) + 255) // 256

        for pass_id in (0, 1, 2):
            if pass_id > 0:
                self.barrier()
            push = struct.pack("IIIII", batch_size, seq_len, features, pass_id, features)
            wg = workgroups_bs if pass_id < 2 else workgroups_flat
            self.dispatch(pipeline, pipeline_layout, descriptor_set, (wg,), push)
        self.barrier()

        return ChainBufferHandle(buf_out, data_nbytes, shape)

    def read_multiple(self, handles: list[ChainBufferHandle]) -> list[np.ndarray]:
        """Submit once, wait, download every handle to numpy (same order as *handles*), then release chain-owned buffers.

        Use for MoE fan-out: several ``linear`` / other ops recorded, one fence wait, multiple CPU reads.
        """
        if not self._recording:
            raise RuntimeError(
                "read_multiple() must be called before exiting the chain_record context"
            )
        self.submit_and_wait()
        self._submitted = True
        out: list[np.ndarray] = []
        for h in handles:
            arr = self._fnn._download_buffer(h.buf, h.nbytes, np.float32)
            out.append(arr.reshape(h.shape))
        self._release_owned()
        return out

    def read(self, h: ChainBufferHandle) -> np.ndarray:
        """Submit recorded commands, wait, download *h* to numpy, release chain-owned buffers."""
        return self.read_multiple([h])[0]

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._submitted and self._recording:
            try:
                self.submit_and_wait()
            except Exception:
                pass
        if not self._released:
            self._release_owned()
        return CommandRecorder.__exit__(self, exc_type, exc_val, exc_tb)
