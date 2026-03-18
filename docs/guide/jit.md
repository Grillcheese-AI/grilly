# JIT Compilation

Grilly's JIT traces a sequence of GPU operations during the first forward pass, then replays them as a fused batch on subsequent calls. Equivalent to `torch.jit.trace`.

---

## @grilly.jit Decorator

```python
import grilly
import grilly.functional as F

@grilly.jit
def forward(x):
    h = F.linear(x, w)
    h = F.relu(h)
    return F.linear(h, w2)

# First call: traces ops, builds TracedGraph
y = forward(x)

# Subsequent calls: replays fused graph (faster)
y = forward(x)
```

The decorator captures the op graph for fixed-shape inputs. On replay, all ops are dispatched as a single `CommandBatch` submission instead of individual dispatches.

---

## Explicit Tracing

For more control, trace a model explicitly:

```python
from grilly.backend.jit import trace

traced = trace(model, example_input)
y = traced(new_input)
```

---

## TracedGraph

The trace produces a `TracedGraph` that records the sequence of operations and their data flow:

```python
from grilly.backend.jit import Tracer

tracer = Tracer()
with tracer:
    tracer.register_input(x)
    y = model(x)
    graph = tracer.build([y])

print(graph.summary())
# TracedGraph: 5 ops
#   [0] linear: [0, 1] -> 2 (32, 256)
#   [1] relu: [2] -> 3 (32, 256)
#   [2] linear: [3, 4] -> 5 (32, 10)
#   ...
```

---

## How It Works

1. **Trace phase**: The `Tracer` singleton intercepts all bridge function calls via `_maybe_trace()`. Each op records its name, input tensor IDs, output tensor ID, and output shape as an `OpRecord`.

2. **Build phase**: `tracer.build()` assembles the `OpRecord` list into a `TracedGraph` with explicit input/output mappings.

3. **Replay phase**: The graph replays ops in sequence. With the C++ backend, this can batch multiple dispatches into a single `CommandBatch` Vulkan submission.

---

## Limitations

- **Fixed shapes only**: The trace is shape-specific. Different input shapes require a new trace.
- **No control flow**: Branches (`if`/`else`) are captured as-executed. The same branch always runs on replay.
- **Single active trace**: You cannot nest JIT traces.

---

## OpGraph (C++ Side)

The C++ backend (`cpp/src/op_graph.cpp`) implements a native `OpGraph` for fusing sequences of Vulkan dispatches at the command buffer level. The Python `TracedGraph` maps to this native structure when the C++ backend is available.
