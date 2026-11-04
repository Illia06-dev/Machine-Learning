import json
import math
import os
import random
import statistics
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import make_moons

warnings.filterwarnings("ignore", category=FutureWarning)

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(RESULTS, exist_ok=True)

lines = []


def log(s=""):
    print(s)
    lines.append(str(s))


class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data:.6f}, grad={self.grad:.6f})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, k):
        out = Value(self.data ** k, (self,), f"**{k}")

        def _backward():
            self.grad += k * self.data ** (k - 1) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self,), "ReLU")

        def _backward():
            self.grad += (1.0 if out.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + math.exp(-self.data))
        out = Value(s, (self,), "sigmoid")

        def _backward():
            self.grad += s * (1.0 - s) * out.grad

        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * other ** -1

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return (-self) + other

    def __rmul__(self, other):
        return self * other

    def __rtruediv__(self, other):
        return Value(other) * self ** -1

    def backward(self):
        topo, visited = [], set()
        stack = [(self, False)]
        while stack:
            v, done = stack.pop()
            if done:
                topo.append(v)
                continue
            if v in visited:
                continue
            visited.add(v)
            stack.append((v, True))
            for p in v._prev:
                if p not in visited:
                    stack.append((p, False))

        self.grad = 1.0
        for v in reversed(topo):
            v._backward()


def vsum(values):
    total = Value(0.0)
    for v in values:
        total = total + v
    return total


def gradient_check():
    log("=== Task 4: gradient check ===")
    N = 5
    random.seed(0)
    a_data = [random.uniform(-2, 2) for _ in range(N)]
    b_data = [random.uniform(-2, 2) for _ in range(N)]
    c_data = [random.uniform(-2, 2) for _ in range(N)]

    a = [Value(x) for x in a_data]
    b = [Value(x) for x in b_data]
    c = [Value(x) for x in c_data]
    out = vsum([((ai * bi + ci) ** 2).relu() for ai, bi, ci in zip(a, b, c)])
    out.backward()
    g_ours = {"a": [v.grad for v in a], "b": [v.grad for v in b], "c": [v.grad for v in c]}

    ta = torch.tensor(a_data, dtype=torch.float64, requires_grad=True)
    tb = torch.tensor(b_data, dtype=torch.float64, requires_grad=True)
    tc = torch.tensor(c_data, dtype=torch.float64, requires_grad=True)
    tout = ((ta * tb + tc) ** 2).relu().sum()
    tout.backward()
    g_torch = {"a": ta.grad.tolist(), "b": tb.grad.tolist(), "c": tc.grad.tolist()}

    def f_plain(a, b, c):
        return sum(max((ai * bi + ci) ** 2, 0.0) for ai, bi, ci in zip(a, b, c))

    h = 1e-6
    data = {"a": a_data, "b": b_data, "c": c_data}
    g_num = {}
    for name in data:
        grads = []
        for i in range(N):
            plus = {k: list(v) for k, v in data.items()}
            minus = {k: list(v) for k, v in data.items()}
            plus[name][i] += h
            minus[name][i] -= h
            grads.append((f_plain(**plus) - f_plain(**minus)) / (2 * h))
        g_num[name] = grads

    def max_diff(g1, g2):
        return max(abs(x - y) for k in g1 for x, y in zip(g1[k], g2[k]))

    d_torch = max_diff(g_ours, g_torch)
    d_num = max_diff(g_ours, g_num)
    log(f"forward:  ours={out.data:.10f}  torch={tout.item():.10f}")
    for k in "abc":
        log(f"d/d{k} ours : {[round(x, 6) for x in g_ours[k]]}")
        log(f"d/d{k} torch: {[round(x, 6) for x in g_torch[k]]}")
    log(f"max |ours - torch|     = {d_torch:.3e}")
    log(f"max |ours - numerical| = {d_num:.3e}")
    assert d_torch < 1e-6 and d_num < 1e-6
    log("PASSED: both differences < 1e-6")

    with open(os.path.join(RESULTS, "grad_check.json"), "w") as fh:
        json.dump({"max_abs_diff_vs_torch": d_torch, "max_abs_diff_vs_numerical": d_num,
                   "ours": g_ours, "torch": g_torch, "numerical": g_num}, fh, indent=2)


N_SAMPLES, EPOCHS, LR, RUNS = 100, 100, 0.5, 5
EPS = 1e-7
X, y = make_moons(n_samples=N_SAMPLES, noise=0.1, random_state=0)


class Neuron:
    def __init__(self, n_in, act=True):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_in)]
        self.b = Value(0.0)
        self.act = act

    def __call__(self, x):
        z = vsum([wi * xi for wi, xi in zip(self.w, x)]) + self.b
        return z.relu() if self.act else z

    def parameters(self):
        return self.w + [self.b]


class MLP:
    def __init__(self):
        self.hidden = [Neuron(2) for _ in range(8)]
        self.out = Neuron(8, act=False)

    def __call__(self, x):
        h = [n(x) for n in self.hidden]
        return self.out(h).sigmoid()

    def parameters(self):
        return [p for n in self.hidden for p in n.parameters()] + self.out.parameters()


def train_engine(seed):
    random.seed(seed)
    model = MLP()
    losses = []
    for _ in range(EPOCHS):
        preds = [model(list(map(float, xi))) for xi in X]
        bce = [-(yi * (p + EPS).log() + (1 - yi) * (1 - p + EPS).log())
               for p, yi in zip(preds, y)]
        loss = vsum(bce) * (1.0 / len(bce))
        for p in model.parameters():
            p.grad = 0.0
        loss.backward()
        for p in model.parameters():
            p.data -= LR * p.grad
        losses.append(loss.data)
    acc = np.mean([(model(list(map(float, xi))).data > 0.5) == yi for xi, yi in zip(X, y)])
    return losses, acc


def train_torch(seed):
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.ReLU(),
                                torch.nn.Linear(8, 1), torch.nn.Sigmoid())
    opt = torch.optim.SGD(model.parameters(), lr=LR)
    loss_fn = torch.nn.BCELoss()
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    losses = []
    for _ in range(EPOCHS):
        loss = loss_fn(model(Xt), yt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    with torch.no_grad():
        acc = ((model(Xt) > 0.5).float() == yt).float().mean().item()
    return losses, acc


def bench(fn):
    times, last = [], None
    for run in range(RUNS):
        t0 = time.perf_counter()
        last = fn(seed=run)
        times.append(time.perf_counter() - t0)
    return times, last


def train_moons():
    log("\n=== Task 5: make_moons 2-8-1 ===")
    t_eng, (loss_eng, acc_eng) = bench(train_engine)
    t_tor, (loss_tor, acc_tor) = bench(train_torch)
    med_eng, med_tor = statistics.median(t_eng), statistics.median(t_tor)

    log(f"Engine : median {med_eng:.3f} s | final loss {loss_eng[-1]:.4f} | acc {acc_eng:.2f}")
    log(f"PyTorch: median {med_tor:.3f} s | final loss {loss_tor[-1]:.4f} | acc {acc_tor:.2f}")
    log(f"PyTorch is {med_eng / med_tor:.1f}x faster")

    with open(os.path.join(RESULTS, "moons_timing.json"), "w") as fh:
        json.dump({"epochs": EPOCHS, "n_samples": N_SAMPLES, "runs": RUNS,
                   "engine_times_s": t_eng, "torch_times_s": t_tor,
                   "engine_median_s": med_eng, "torch_median_s": med_tor,
                   "speedup": med_eng / med_tor,
                   "engine_final_loss": loss_eng[-1], "torch_final_loss": loss_tor[-1],
                   "engine_acc": float(acc_eng), "torch_acc": acc_tor}, fh, indent=2)

    plt.plot(loss_eng, label="our engine")
    plt.plot(loss_tor, label="PyTorch")
    plt.xlabel("epoch")
    plt.ylabel("BCE loss")
    plt.title("make_moons, 2-8-1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "moons_loss.png"), dpi=120)
    plt.close()


def timeit(fn, reps=50, warmup=10):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e3


def walk(fn, depth=0, seen=None):
    seen = seen if seen is not None else set()
    if fn is None or fn in seen:
        return
    seen.add(fn)
    log("  " * depth + type(fn).__name__)
    for nxt, _ in fn.next_functions:
        walk(nxt, depth + 1, seen)


def torch_inspect():
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(64, 256), torch.nn.ReLU(),
                                torch.nn.Linear(256, 256), torch.nn.ReLU(),
                                torch.nn.Linear(256, 1)).eval()
    x = torch.randn(512, 64)

    log("\n=== Task 6: autograd graph (grad_fn) ===")
    walk(model(x).sum().grad_fn)

    log("\n=== torch.fx graph ===")
    log(torch.fx.symbolic_trace(model).graph)

    log("=== eager vs compiled (after warm-up) ===")
    res = {"runs": []}
    with torch.no_grad():
        try:
            t0 = time.perf_counter()
            compiled = torch.compile(model)
            compiled(x)
            res["compile_time_s"] = time.perf_counter() - t0
            log(f"torch.compile first call: {res['compile_time_s']:.1f} s")
        except Exception as e:
            log(f"torch.compile unavailable ({type(e).__name__}) -> only jit.trace")
            compiled = None
        traced = torch.jit.trace(model, x)

        for batch in (8, 512, 8192):
            xb = torch.randn(batch, 64)
            row = {"batch": batch, "eager_ms": timeit(lambda: model(xb)),
                   "trace_ms": timeit(lambda: traced(xb))}
            line = f"batch {batch:5d}: eager {row['eager_ms']:.3f} ms | jit.trace {row['trace_ms']:.3f} ms"
            if compiled is not None:
                row["compile_ms"] = timeit(lambda: compiled(xb))
                row["max_abs_diff"] = (compiled(xb) - model(xb)).abs().max().item()
                line += f" | torch.compile {row['compile_ms']:.3f} ms (max diff {row['max_abs_diff']:.1e})"
            res["runs"].append(row)
            log(line)

    log("\n=== tensor inspection ===")
    a = torch.randn(4096, 4096)
    at = a.t()
    at_c = at.contiguous()
    for name, t in [("a", a), ("a.t()", at), ("a.t().contiguous()", at_c)]:
        log(f"{name:20s} shape={tuple(t.shape)} dtype={t.dtype} stride={t.stride()} "
            f"contiguous={t.is_contiguous()} same_storage={t.data_ptr() == a.data_ptr()}")

    cont = {
        "contiguous_add_ms": timeit(lambda: a + 1.0, reps=20, warmup=3),
        "noncontiguous_add_ms": timeit(lambda: at + 1.0, reps=20, warmup=3),
        "contiguous_rowsum_ms": timeit(lambda: at_c.sum(dim=1), reps=20, warmup=3),
        "noncontiguous_rowsum_ms": timeit(lambda: at.sum(dim=1), reps=20, warmup=3),
        "contiguous_copy_ms": timeit(lambda: at.contiguous(), reps=20, warmup=3),
        "mixed_contig_ms": timeit(lambda: a + at_c, reps=20, warmup=3),
        "mixed_noncontig_ms": timeit(lambda: a + at, reps=20, warmup=3),
    }
    log(f"a + 1        (contiguous)     : {cont['contiguous_add_ms']:.2f} ms")
    log(f"a.t() + 1    (non-contiguous) : {cont['noncontiguous_add_ms']:.2f} ms")
    log(f"sum(dim=1)   (contiguous)     : {cont['contiguous_rowsum_ms']:.2f} ms")
    log(f"sum(dim=1)   (non-contiguous) : {cont['noncontiguous_rowsum_ms']:.2f} ms")
    log(f".contiguous() copy            : {cont['contiguous_copy_ms']:.2f} ms")
    log(f"a + a.t().contiguous()        : {cont['mixed_contig_ms']:.2f} ms")
    log(f"a + a.t()    (mixed layouts)  : {cont['mixed_noncontig_ms']:.2f} ms")

    res.update(cont)
    res["torch_version"] = torch.__version__
    with open(os.path.join(RESULTS, "torch_inspect.json"), "w") as fh:
        json.dump(res, fh, indent=2)


if __name__ == "__main__":
    gradient_check()
    train_moons()
    torch_inspect()
    with open(os.path.join(RESULTS, "output.txt"), "w") as fh:
        fh.write("\n".join(lines))