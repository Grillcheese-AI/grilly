# Autograd

Grilly provides automatic differentiation through the `Variable` class. Operations build a computation graph during the forward pass; calling `.backward()` traverses the graph in reverse to compute gradients.

---

## Variable

`Variable` wraps a numpy array and tracks operations for gradient computation:

```python
from grilly.nn import Variable, tensor

x = Variable(tensor([1.0, 2.0, 3.0]), requires_grad=True)
y = (x * x).sum()
y.backward()
print(x.grad)  # [2.0, 4.0, 6.0]
```

Set `requires_grad=True` to track gradients. Intermediate results inherit gradient tracking from their inputs.

---

## Operator Overloading

`Variable` supports natural Python syntax through operator overloading:

```python
a = Variable(tensor([2.0, 3.0]), requires_grad=True)
b = Variable(tensor([4.0, 5.0]), requires_grad=True)

# Arithmetic
c = a + b
c = a - b
c = a * b
c = a / b
c = a ** 2

# Reductions
s = c.sum()
m = c.mean()

# Chain operations
loss = ((a * b - 1.0) ** 2).mean()
loss.backward()
print(a.grad)
print(b.grad)
```

---

## Supported Operations

### Arithmetic

`add`, `sub`, `mul`, `div`, `neg`, `pow`, `matmul`

### Activations

`relu`, `sigmoid`, `tanh`, `gelu`, `silu`, `leaky_relu`, `elu`, `softplus`, `softmax`

### Reductions

`sum`, `mean`, `max`, `min`, `var`, `std`, `norm`

### Shape Operations

`reshape`, `transpose`, `squeeze`, `unsqueeze`, `flatten`, `view`, `expand`, `repeat`, `permute`, `contiguous`, `clone`, `concat`, `stack`

### Trigonometric

`sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`

### Math

`exp`, `log`, `sqrt`, `abs`, `clamp`

### Comparisons

`eq`, `ne`, `lt`, `le`, `gt`, `ge`, `where`

### Loss Functions

`cross_entropy`, `mse_loss`, `l1_loss`, `smooth_l1_loss`, `bce_loss`, `bce_with_logits_loss`, `nll_loss`, `kl_div_loss`

---

## Context Managers

```python
from grilly.nn import no_grad, enable_grad, is_grad_enabled

# Disable gradient tracking (inference, evaluation)
with no_grad():
    output = model(x)
    # No computation graph built

# Re-enable inside a no_grad block
with no_grad():
    with enable_grad():
        y = Variable(tensor([1.0]), requires_grad=True)
        z = y * 2
        z.backward()  # Works
```

---

## Factory Functions

```python
from grilly.nn import tensor, zeros, ones, randn, rand, linspace, arange, eye, full

x = tensor([1.0, 2.0, 3.0])
z = zeros(3, 4)
o = ones(2, 3)
r = randn(5, 5)
e = eye(4)
```

---

## Custom Functions

Extend autograd with custom forward/backward operations:

```python
from grilly.nn import Function, FunctionCtx

class MyReLU(Function):
    @staticmethod
    def forward(ctx: FunctionCtx, x):
        ctx.save_for_backward(x)
        return x * (x > 0)

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output):
        x, = ctx.saved_tensors
        return grad_output * (x > 0)

# Use it
x = Variable(tensor([-1.0, 0.5, 2.0]), requires_grad=True)
y = MyReLU.apply(x)
y.sum().backward()
print(x.grad)  # [0.0, 1.0, 1.0]
```

---

## Backend Autograd (GradientTape)

The backend also provides a lower-level `GradientTape` for recording operations at the Vulkan dispatch level:

```python
from grilly.backend.autograd_core import GradientTape

with GradientTape() as tape:
    tape.watch(x)
    y = backend.fnn.linear(x, w)
    grads = tape.gradient(y, x)
```

This is used internally by `nn.Module.backward()`. Most users should use `Variable` instead.
