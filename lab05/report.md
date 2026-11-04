# Lab 5 — Frameworks and automatic differentiation

## 1–3. Autograd engine
- `Value` stores `data`, `grad`, parent nodes `_prev`, op name `_op` and a local
  derivative rule `_backward`, created during the forward pass.
- Ops: `+`, `*`, `**k`, `relu` (+ `sigmoid`, `log` for BCE; `-` and `/` are built
  from them). Every op creates a node linked to its parents, so the forward pass
  builds the computation graph.
- `backward()`: topological sort (iterative DFS) → output grad = 1 → nodes visited in
  reverse topological order → every rule uses `+=`, so gradients are accumulated.

## 4. Gradient check
f = ((a*b + c)**2).relu().sum(), vectors of length 5, float64.
Forward value: ours = torch = 4.5114164911.

| comparison | max abs difference |
|---|---|
| ours vs PyTorch autograd | 0.0e+00 |
| ours vs central differences (h=1e-6) | 3.8e-10 |

Both < 1e-6 → PASSED. Raw values: `results/grad_check.json`.

## 5. make_moons, 2 → 8 → 1
ReLU hidden, sigmoid output, BCE, full-batch SGD lr=0.5, 100 samples, 100 epochs,
median of 5 runs.

| implementation | median time | final loss | accuracy |
|---|---|---|---|
| our engine | 9.027 s | 0.2577 | 0.88 |
| PyTorch | 0.028 s | 0.2504 | 0.89 |

PyTorch is ~320× faster: our engine creates a Python object for every scalar
operation and runs Python closures in backward, while PyTorch runs one vectorised
C++ kernel per layer. Both reach almost the same loss and accuracy.
Loss curves: `results/moons_loss.png`.

## 6. PyTorch graph, execution modes, contiguity
Model: Linear(64,256)-ReLU-Linear(256,256)-ReLU-Linear(256,1).

**Graph.** `grad_fn` chain: `SumBackward0 → AddmmBackward0 → ReluBackward0 →
AddmmBackward0 → ...`; leaves are `AccumulateGrad` (parameters), `TBackward0` is the
weight transpose in Linear. `torch.fx.symbolic_trace` shows the same 5 modules as a
static graph. Full output: `results/output.txt`.

**Eager vs traced (after warm-up, median of 50 calls, CPU).**
`torch.compile` was unavailable on this Windows machine (`InductorError`: no C++
compiler for the Inductor backend), so `torch.jit.trace` was used, as allowed by
the task.

| batch | eager | jit.trace |
|---|---|---|
| 8 | 0.047 ms | 0.039 ms |
| 512 | 0.311 ms | 0.323 ms |
| 8192 | 4.512 ms | 4.668 ms |

Tracing is ~17% faster only for batch 8, where Python overhead per call matters.
For batch 512 and 8192 the time is dominated by matmul kernels, which are the same
in both modes, so there is no gain (differences are within noise).

**Shape / dtype / contiguity.** `a`: (4096,4096), float32, stride (4096,1),
contiguous. `a.t()` is a view: same storage, stride (1,4096), not contiguous.
`.contiguous()` creates a copy with stride (4096,1).

| operation | contiguous | non-contiguous |
|---|---|---|
| `x + 1` | 10.11 ms | 10.39 ms |
| `x.sum(dim=1)` | 3.17 ms | 6.93 ms |
| `a + x` (mixing layouts) | 13.13 ms | 64.76 ms |
| `.contiguous()` copy | — | 26.01 ms |

An element-wise op on a single transposed view costs the same, but a reduction over a
strided dimension is ~2.2× slower and mixing two layouts is ~4.9× slower, because
memory is read with jumps of 4096 floats and cache locality is lost.

## When compilation helps
Compilation and tracing remove per-op Python overhead and (for torch.compile) fuse
element-wise ops into fewer kernels. They help when the same model runs many times
with fixed shapes (training loops, inference serving), when the model consists of
many small or element-wise ops, and when the one-time compile cost is amortised.
They do not help much when time is dominated by large matmuls (our batch 512 and 8192:
no gain), for one-off runs, or when shapes and Python control flow change often
(recompilation, graph breaks; tracing records only one branch of an `if`). In our
test the only gain was for the smallest batch (8), where Python overhead is a large
part of the runtime. On GPU and bigger models the benefit is usually larger.
For non-contiguous tensors, call `.contiguous()` once before repeated heavy
operations instead of paying for strided access every time.