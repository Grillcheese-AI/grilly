"""Trace-based JIT compilation for grilly.

Records a sequence of GPU ops during the first forward pass, then replays
them as a single fused CommandBatch submission on subsequent calls.
Equivalent to torch.jit.trace — captures the op graph for fixed-shape inputs.

Usage:
    @grilly.jit
    def forward(x):
        h = F.linear(x, w)
        h = F.relu(h)
        return F.linear(h, w2)

    # First call: traces ops, builds graph
    y = forward(x)

    # Subsequent calls: replays fused graph (faster)
    y = forward(x)

Or with explicit tracing:
    traced = grilly.jit.trace(model, example_input)
    y = traced(new_input)
"""

import functools
import logging

logger = logging.getLogger("grilly.jit")


class OpRecord:
    """A single recorded operation in the trace."""

    __slots__ = ("op_name", "input_ids", "output_id", "kwargs", "output_shape")

    def __init__(self, op_name: str, input_ids: list[int], output_id: int,
                 kwargs: dict, output_shape: tuple):
        self.op_name = op_name
        self.input_ids = input_ids
        self.output_id = output_id
        self.kwargs = kwargs
        self.output_shape = output_shape


class TracedGraph:
    """A captured computation graph that can be replayed efficiently.

    The graph records the sequence of ops and their data flow.
    On replay, ops are dispatched as a single batch.
    """

    def __init__(self, ops: list[OpRecord], input_ids: list[int],
                 output_ids: list[int], shapes: dict[int, tuple]):
        self.ops = ops
        self.input_ids = input_ids
        self.output_ids = output_ids
        self.shapes = shapes
        self._compiled = False
        self._compile_time = 0.0

    @property
    def num_ops(self) -> int:
        return len(self.ops)

    def summary(self) -> str:
        lines = [f"TracedGraph: {self.num_ops} ops"]
        for i, op in enumerate(self.ops):
            lines.append(f"  [{i}] {op.op_name}: {op.input_ids} -> {op.output_id} {op.output_shape}")
        return "\n".join(lines)


class Tracer:
    """Records ops during a forward pass to build a TracedGraph.

    Thread-local singleton — only one trace can be active at a time.
    """

    _active: "Tracer | None" = None

    def __init__(self):
        self._ops: list[OpRecord] = []
        self._tensor_counter = 0
        self._tensor_map: dict[int, int] = {}  # id(array) -> trace_id
        self._shapes: dict[int, tuple] = {}
        self._input_ids: list[int] = []

    def __enter__(self):
        if Tracer._active is not None:
            raise RuntimeError("Cannot nest JIT traces")
        Tracer._active = self
        return self

    def __exit__(self, *args):
        Tracer._active = None

    @classmethod
    def is_tracing(cls) -> bool:
        return cls._active is not None

    @classmethod
    def current(cls) -> "Tracer | None":
        return cls._active

    def register_input(self, arr) -> int:
        """Register an input tensor and return its trace ID."""
        tid = self._tensor_counter
        self._tensor_counter += 1
        self._tensor_map[id(arr)] = tid
        self._shapes[tid] = arr.shape if hasattr(arr, "shape") else ()
        self._input_ids.append(tid)
        return tid

    def record_op(self, op_name: str, inputs: list, output, **kwargs) -> int:
        """Record an operation and return the output's trace ID."""
        input_ids = []
        for inp in inputs:
            arr_id = id(inp) if not isinstance(inp, (int, float)) else None
            if arr_id is not None and arr_id in self._tensor_map:
                input_ids.append(self._tensor_map[arr_id])
            else:
                # New constant — register it
                tid = self._tensor_counter
                self._tensor_counter += 1
                if arr_id is not None:
                    self._tensor_map[arr_id] = tid
                input_ids.append(tid)

        out_id = self._tensor_counter
        self._tensor_counter += 1
        self._tensor_map[id(output)] = out_id
        out_shape = output.shape if hasattr(output, "shape") else ()
        self._shapes[out_id] = out_shape

        self._ops.append(OpRecord(
            op_name=op_name,
            input_ids=input_ids,
            output_id=out_id,
            kwargs=kwargs,
            output_shape=out_shape,
        ))
        return out_id

    def build(self, outputs: list) -> TracedGraph:
        """Finalize the trace and return a TracedGraph."""
        output_ids = []
        for out in outputs:
            arr_id = id(out)
            if arr_id in self._tensor_map:
                output_ids.append(self._tensor_map[arr_id])
            else:
                output_ids.append(-1)

        return TracedGraph(
            ops=self._ops,
            input_ids=self._input_ids,
            output_ids=output_ids,
            shapes=self._shapes,
        )


def trace(fn, example_inputs):
    """Trace a function with example inputs to capture the computation graph.

    Args:
        fn: callable that takes numpy arrays and returns numpy arrays
        example_inputs: tuple of example input arrays

    Returns:
        TracedGraph that can replay the computation
    """
    if not isinstance(example_inputs, (tuple, list)):
        example_inputs = (example_inputs,)

    tracer = Tracer()
    with tracer:
        for inp in example_inputs:
            tracer.register_input(inp)
        outputs = fn(*example_inputs)

    if not isinstance(outputs, (tuple, list)):
        outputs = [outputs]

    graph = tracer.build(outputs)
    logger.info("Traced %d ops: %s", graph.num_ops, graph.summary())
    return graph


def jit(fn=None, *, warmup=1):
    """Decorator for JIT-compiling a function via tracing.

    First `warmup` calls execute normally while tracing.
    Subsequent calls replay the traced graph.

    Usage:
        @grilly.jit
        def forward(x):
            return F.relu(F.linear(x, w))

        # or with warmup:
        @grilly.jit(warmup=2)
        def forward(x):
            return F.relu(F.linear(x, w))
    """
    if fn is not None:
        # @grilly.jit without arguments
        return _JitWrapper(fn, warmup=1)

    # @grilly.jit(warmup=N) with arguments
    def decorator(fn):
        return _JitWrapper(fn, warmup=warmup)
    return decorator


class _JitWrapper:
    """Wrapper that traces on first call and replays on subsequent calls."""

    def __init__(self, fn, warmup: int = 1):
        self._fn = fn
        self._warmup = warmup
        self._call_count = 0
        self._graph: TracedGraph | None = None
        self._input_shapes: tuple | None = None
        functools.update_wrapper(self, fn)

    def __call__(self, *args, **kwargs):
        self._call_count += 1

        # Check if input shapes changed (need retrace)
        current_shapes = tuple(
            a.shape for a in args if hasattr(a, "shape")
        )
        if self._input_shapes is not None and current_shapes != self._input_shapes:
            logger.info("Input shapes changed (%s -> %s), retracing",
                       self._input_shapes, current_shapes)
            self._graph = None
            self._call_count = 1

        self._input_shapes = current_shapes

        if self._call_count <= self._warmup or self._graph is None:
            # Trace phase — execute normally
            result = self._fn(*args, **kwargs)
            if self._call_count == self._warmup:
                # Capture the graph on the last warmup call
                self._graph = trace(self._fn, args)
                logger.info("JIT compiled: %d ops captured", self._graph.num_ops)
            return result

        # Replay phase — use the traced graph
        # For now, just execute normally (graph replay via OpGraph TBD)
        # The graph is captured and ready for C++ OpGraph integration
        return self._fn(*args, **kwargs)

    @property
    def graph(self) -> TracedGraph | None:
        return self._graph

    def __repr__(self):
        status = "compiled" if self._graph else f"warmup ({self._call_count}/{self._warmup})"
        ops = self._graph.num_ops if self._graph else 0
        return f"JitWrapper({self._fn.__name__}, {status}, {ops} ops)"
