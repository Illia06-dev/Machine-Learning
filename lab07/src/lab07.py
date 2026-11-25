import copy
import io
import os
import platform
import random
import statistics
import time
from importlib.metadata import version as pkg_version

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

SEED = 42
RESULTS = "results"
MODELS = os.path.join(RESULTS, "models")
os.makedirs(MODELS, exist_ok=True)


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def print_versions():
    lines = [f"python=={platform.python_version()}"]
    for p in ["torch", "torchvision", "numpy", "pandas", "matplotlib"]:
        lines.append(f"{p}=={pkg_version(p)}")
    with open(os.path.join(RESULTS, "versions.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def cnn(c1, c2, hidden):
    return nn.Sequential(
        nn.Conv2d(1, c1, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(c1, c2, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(c2 * 7 * 7, hidden), nn.ReLU(), nn.Linear(hidden, 10))


def get_data():
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
    train = datasets.FashionMNIST("../data", train=True, download=True, transform=tf)
    test = datasets.FashionMNIST("../data", train=False, download=True, transform=tf)
    return train, test


def train(model, train_ds, epochs, lr=1e-3, teacher=None, T=4.0, alpha=0.5):
    set_seed()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    data = DataLoader(train_ds, batch_size=128, shuffle=True, generator=torch.Generator().manual_seed(SEED))
    for _ in range(epochs):
        model.train()
        for x, y in data:
            out = model(x)
            loss = F.cross_entropy(out, y)
            if teacher is not None:
                with torch.no_grad():
                    t_out = teacher(x)
                kl = F.kl_div(F.log_softmax(out / T, 1), F.softmax(t_out / T, 1), reduction="batchmean")
                loss = alpha * loss + (1 - alpha) * T * T * kl
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model.eval()


@torch.no_grad()
def accuracy(model, test_loader):
    model.eval()
    return sum((model(x).argmax(1) == y).sum().item() for x, y in test_loader) / len(test_loader.dataset)


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def size_kb(state_dict):
    buf = io.BytesIO()
    torch.save(state_dict, buf)
    return len(buf.getvalue()) / 1024


@torch.no_grad()
def latency_ms(model, x, warmup=10, runs=100):
    model.eval()
    for _ in range(warmup):
        model(x)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model(x)
        times.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(times)


def prunable(model):
    return [(m, "weight") for m in model if isinstance(m, (nn.Conv2d, nn.Linear))]


def sparsity(model):
    total = zeros = 0
    for m, _ in prunable(model):
        total += m.weight.numel()
        zeros += int((m.weight == 0).sum())
    return zeros / total


def macs(model, x):
    counts = {"dense": 0, "nonzero": 0}

    def hook(m, inp, out):
        positions = out[0, 0].numel() if isinstance(m, nn.Conv2d) else 1
        counts["dense"] += positions * m.weight.numel()
        counts["nonzero"] += positions * int((m.weight != 0).sum())

    handles = [m.register_forward_hook(hook) for m, _ in prunable(model)]
    with torch.no_grad():
        model(x)
    for h in handles:
        h.remove()
    return counts


# ===== Task 2 =====
def prune_and_finetune(baseline, train_ds, method, amount):
    model = copy.deepcopy(baseline)
    params = prunable(model)
    if method == "global":
        prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=amount)
    elif method == "layer-wise":
        for m, name in params:
            prune.l1_unstructured(m, name, amount=amount)
    elif method == "structured":
        for m in model:
            if isinstance(m, nn.Conv2d):
                prune.ln_structured(m, "weight", amount=amount, n=2, dim=0)
                kept = (m.weight_mask.flatten(1).sum(1) > 0).float()
                prune.custom_from_mask(m, "bias", kept)
    train(model, train_ds, epochs=3, lr=1e-4)
    for m in model:
        for name in ("weight", "bias"):
            if prune.is_pruned(m) and hasattr(m, f"{name}_mask"):
                prune.remove(m, name)
    return model


def remove_filters(model):
    conv1, conv2, fc1, fc2 = model[0], model[3], model[7], model[9]
    keep1 = (conv1.weight.flatten(1).abs().sum(1) > 0).nonzero().flatten()
    keep2 = (conv2.weight.flatten(1).abs().sum(1) > 0).nonzero().flatten()
    small = cnn(len(keep1), len(keep2), fc1.out_features)
    small[0].weight.data, small[0].bias.data = conv1.weight[keep1].clone(), conv1.bias[keep1].clone()
    small[3].weight.data = conv2.weight[keep2][:, keep1].clone()
    small[3].bias.data = conv2.bias[keep2].clone()
    features = (keep2[:, None] * 49 + torch.arange(49)).flatten()
    small[7].weight.data, small[7].bias.data = fc1.weight[:, features].clone(), fc1.bias.clone()
    small[9].load_state_dict(fc2.state_dict())
    return small.eval()


# ===== Task 3 =====
def sparse_state_dict(model):
    return {k: v.reshape(v.shape[0], -1).to_sparse() if k.endswith("weight") else v
            for k, v in model.state_dict().items()}


def main():
    set_seed()
    print_versions()
    train_ds, test_ds = get_data()
    test_loader = DataLoader(test_ds, batch_size=1000)
    x1 = test_ds[0][0].unsqueeze(0)

    # ===== Task 1 =====
    print("\n=== Task 1 ===")
    set_seed()
    baseline = train(cnn(32, 64, 300), train_ds, epochs=5)
    torch.save(baseline.state_dict(), os.path.join(MODELS, "baseline.pt"))
    base = {"accuracy": accuracy(baseline, test_loader), "params": n_params(baseline),
            "size_kb": size_kb(baseline.state_dict()), "latency_ms": latency_ms(baseline, x1)}
    print(base)

    # ===== Task 2 =====
    print("\n=== Task 2 ===")
    rows, pruned_models = [], {}
    dense = prune_and_finetune(baseline, train_ds, "none", 0.0)
    for method, amounts in [("global", [0.25, 0.5, 0.75, 0.9, 0.95]),
                            ("layer-wise", [0.25, 0.5, 0.75, 0.9, 0.95]),
                            ("structured", [0.1, 0.25, 0.5])]:
        pruned_models[(method, 0.0)] = dense
        rows.append({"method": method, "target": 0.0, "sparsity": 0.0, "accuracy": accuracy(dense, test_loader)})
        for amount in amounts:
            m = prune_and_finetune(baseline, train_ds, method, amount)
            pruned_models[(method, amount)] = m
            rows.append({"method": method, "target": amount, "sparsity": sparsity(m),
                         "accuracy": accuracy(m, test_loader)})
            print(rows[-1])
    curves = pd.DataFrame(rows)
    curves.to_csv(os.path.join(RESULTS, "sparsity_curves.csv"), index=False)

    plt.figure(figsize=(7, 4.5))
    for method, label in [("global", "global unstructured"), ("layer-wise", "layer-wise unstructured"),
                          ("structured", "structured (conv filters, L2)")]:
        d = curves[curves.method == method]
        plt.plot(d.target, d.accuracy, "o-", label=label)
    plt.xlabel("pruned fraction (weights, or conv filters for structured)")
    plt.ylabel("test accuracy after 3 epochs fine-tuning")
    plt.title("Accuracy vs sparsity")
    plt.grid(alpha=.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "sparsity_curves.png"), dpi=150)
    torch.save(pruned_models[("global", 0.9)].state_dict(), os.path.join(MODELS, "pruned_global_90.pt"))

    # ===== Task 3 =====
    print("\n=== Task 3 ===")
    storage = []
    for amount in [0.0, 0.25, 0.5, 0.75, 0.9, 0.95]:
        m = pruned_models[("global", amount)]
        storage.append({"global_sparsity": amount, "dense_kb": size_kb(m.state_dict()),
                        "sparse_coo_kb": size_kb(sparse_state_dict(m)), "latency_ms": latency_ms(m, x1)})
    storage = pd.DataFrame(storage)
    storage.to_csv(os.path.join(RESULTS, "storage_latency.csv"), index=False)
    print(storage.to_string(index=False))

    compact = remove_filters(pruned_models[("structured", 0.5)])
    with torch.no_grad():
        x = next(iter(test_loader))[0]
        same = torch.allclose(compact(x), pruned_models[("structured", 0.5)](x), atol=1e-5)
    print(f"structured 50% with filters physically removed: params {n_params(compact)}, "
          f"latency {latency_ms(compact, x1):.3f} ms, same outputs as masked model: {same}")

    # ===== Task 4 =====
    print("\n=== Task 4 ===")
    set_seed()
    plain = train(cnn(16, 32, 64), train_ds, epochs=5)
    set_seed()
    distilled = train(cnn(16, 32, 64), train_ds, epochs=5, teacher=baseline, T=4.0, alpha=0.5)
    torch.save(distilled.state_dict(), os.path.join(MODELS, "student_distilled.pt"))
    students = pd.DataFrame([{"model": name, "params": n_params(m), "accuracy": accuracy(m, test_loader),
                              "size_kb": size_kb(m.state_dict()), "latency_ms": latency_ms(m, x1)}
                             for name, m in [("teacher (baseline)", baseline), ("student plain", plain),
                                             ("student distilled", distilled)]])
    students.to_csv(os.path.join(RESULTS, "distillation.csv"), index=False)
    print(students.to_string(index=False))

    # ===== Task 5 =====
    print("\n=== Task 5 ===")
    n_train = len(train_ds)
    summary = []
    for name, m, images_seen in [
        ("baseline", baseline, 5 * n_train),
        ("global pruned 90%", pruned_models[("global", 0.9)], 8 * n_train),
        ("structured 50% (filters removed)", compact, 8 * n_train),
        ("student plain", plain, 5 * n_train),
        ("student distilled", distilled, 5 * n_train + 5 * n_train),
    ]:
        c = macs(m, x1)
        summary.append({"model": name,
                        "data: training images processed": images_seen,
                        "model: params": n_params(m),
                        "model: nonzero weights": int(sum(int((p != 0).sum()) for p in m.parameters())),
                        "model: size_kb": size_kb(m.state_dict()),
                        "model: accuracy": accuracy(m, test_loader),
                        "compute: MACs executed": c["dense"],
                        "compute: nonzero MACs": c["nonzero"],
                        "compute: latency_ms": latency_ms(m, x1),
                        "energy: est. uJ per inference": c["dense"] * 4.6e-6})
    summary = pd.DataFrame(summary)
    summary.to_csv(os.path.join(RESULTS, "pillars_summary.csv"), index=False)
    print(summary.T.to_string())


if __name__ == "__main__":
    main()