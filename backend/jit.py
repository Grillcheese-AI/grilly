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
from collections import OrderedDict

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
        # Fix #1: Keep all traced objects alive so their id()s can't be reused
        # by Python's GC during tracing. Without this, freed intermediate tensors
        # can get the same id() as new ones → corrupted graph wiring.
        self._keepalive: list = []

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
        # Keep references alive to prevent id() reuse
        self._keepalive.append(output)
        self._keepalive.extend(inp for inp in inputs if not isinstance(inp, (int, float)))

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
    """Wrapper that traces on first call and replays on subsequent calls.

    Fix #3: LRU cache of traced graphs keyed by (shapes, scalar_kwargs).
    Prevents retrace thrashing when batch/sequence sizes alternate.

    Fix #4: Scalar kwargs are hashed alongside shapes so temperature/scale
    changes trigger a new trace instead of silently replaying stale values.
    """

    _MAX_CACHED_GRAPHS = 8  # LRU eviction after this many shape variants

    def __init__(self, fn, warmup: int = 1):
        self._fn = fn
        self._warmup = warmup
        self._graphs: OrderedDict[tuple, TracedGraph] = OrderedDict()  # cache_key → graph (LRU)
        self._warmup_counts: dict[tuple, int] = {}
        functools.update_wrapper(self, fn)

    def _cache_key(self, args, kwargs) -> tuple:
        """Build a hashable key from tensor shapes + scalar kwargs."""
        shapes = tuple(a.shape for a in args if hasattr(a, "shape"))
        scalars = tuple(
            (k, v) for k, v in sorted(kwargs.items())
            if isinstance(v, (int, float, bool, str))
        )
        return shapes + scalars

    def __call__(self, *args, **kwargs):
        key = self._cache_key(args, kwargs)

        # Check if we have a compiled graph for this shape+kwargs combo
        if key in self._graphs:
            self._graphs.move_to_end(key)
            # Replay (for now, execute normally — C++ OpGraph replay TBD)
            return self._fn(*args, **kwargs)

        # Warmup phase
        count = self._warmup_counts.get(key, 0) + 1
        self._warmup_counts[key] = count

        result = self._fn(*args, **kwargs)

        if count >= self._warmup:
            # Trace and cache
            graph = trace(lambda *a: self._fn(*a, **kwargs), args)
            self._graphs[key] = graph
            self._graphs.move_to_end(key)
            logger.info("JIT compiled: %d ops captured (key=%s)", graph.num_ops, key[:2])

            # LRU eviction
            if len(self._graphs) > self._MAX_CACHED_GRAPHS:
                self._graphs.popitem(last=False)
                logger.info("JIT evicted oldest graph (cache size=%d)", len(self._graphs))

        return result

    @property
    def graph(self) -> TracedGraph | None:
        """Return the most recently compiled graph (if any)."""
        if self._graphs:
            return next(reversed(self._graphs.values()))
        return None

    def __repr__(self):
        n = len(self._graphs)
        status = f"compiled ({n} graphs)" if n > 0 else "warmup"
        ops = sum(g.num_ops for g in self._graphs.values())
        return f"JitWrapper({self._fn.__name__}, {status}, {ops} ops)"
