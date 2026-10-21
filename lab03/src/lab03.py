import platform
import random
import statistics
import sys
import time
from importlib.metadata import version
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn as nn
from thop import profile
from torch.utils.data import DataLoader, random_split
from torchinfo import summary
from torchvision import datasets, transforms

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LAB_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = LAB_DIR.parent / "data"
OUT = LAB_DIR / "results"
OUT.mkdir(exist_ok=True)

SEED = 42
BATCH_SIZE = 128
EPOCHS = 5
LR = 1e-3


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def print_versions():
    print(f"Python {platform.python_version()}")
    for pkg in ["numpy", "matplotlib", "torch", "torchvision", "torchinfo", "thop"]:
        print(f"{pkg}=={version(pkg)}")


def get_loaders():
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
    full_train = datasets.FashionMNIST(DATA_DIR, train=True, download=True, transform=tf)
    test = datasets.FashionMNIST(DATA_DIR, train=False, download=True, transform=tf)
    train, val = random_split(full_train, [55_000, 5_000],
                              generator=torch.Generator().manual_seed(SEED))
    return (DataLoader(train, batch_size=BATCH_SIZE, shuffle=True),
            DataLoader(val, batch_size=BATCH_SIZE),
            DataLoader(test, batch_size=BATCH_SIZE))


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 10),
        )

    def forward(self, x):
        return self.net(x)


class AttentionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(7 * 7, 64)
        layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128,
                                           batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.head = nn.Linear(64, 10)

    def forward(self, x):
        b = x.shape[0]
        patches = x.unfold(2, 7, 7).unfold(3, 7, 7).reshape(b, 16, 49)
        x = self.encoder(self.proj(patches))
        return self.head(x.mean(dim=1))


def build_models():
    return {"MLP": MLP(), "CNN": CNN(), "Attention": AttentionNet()}


def hand_flops():
    linear = lambda n_in, n_out, rows=1: 2 * n_in * n_out * rows
    conv = lambda k, c_in, c_out, h, w: 2 * k * k * c_in * c_out * h * w
    seq, d, d_ff = 16, 64, 128
    encoder_layer = {
        "QKV projections": 2 * seq * d * 3 * d,
        "attention scores": 2 * seq * seq * d,
        "scores x V": 2 * seq * seq * d,
        "output projection": 2 * seq * d * d,
        "feed-forward": 2 * seq * d * d_ff * 2,
    }
    print("Representative layers:")
    print(f"  Linear 784->256:          {linear(784, 256):,}")
    print(f"  Conv2d 16->32 @14x14:     {conv(3, 16, 32, 14, 14):,}")
    for k, v in encoder_layer.items():
        print(f"  Encoder layer, {k:18s} {v:,}")
    print(f"  Encoder layer, total:     {sum(encoder_layer.values()):,}")
    return {
        "MLP": linear(784, 256) + linear(256, 128) + linear(128, 10),
        "CNN": conv(3, 1, 16, 28, 28) + conv(3, 16, 32, 14, 14) + linear(32 * 7 * 7, 10),
        "Attention": linear(49, 64, rows=16) + 2 * sum(encoder_layer.values()) + linear(64, 10),
    }


@torch.inference_mode()
def latency_ms(model, batch, runs):
    model.eval()
    x = torch.randn(batch, 1, 28, 28)
    for _ in range(10):
        model(x)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model(x)
        times.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(times)


def measure_cost(models):
    hand = hand_flops()
    x = torch.randn(1, 1, 28, 28)
    cost = {}
    for name, model in models.items():
        params = summary(model, input_size=(1, 1, 28, 28), verbose=0).total_params
        macs, _, layers = profile(model, inputs=(x,), verbose=False, ret_layer_info=True)
        cost[name] = {
            "params": params,
            "lat_b1": latency_ms(model, 1, 100),
            "lat_b128": latency_ms(model, 128, 20),
        }
        print(f"\n{name}: params={params:,}  hand FLOPs={hand[name]:,}  thop FLOPs={2 * int(macs):,}")
        print(f"  thop MACs per top-level module: { {k: int(v[0]) for k, v in layers.items()} }")
        print(f"  latency: batch 1 = {cost[name]['lat_b1']:.3f} ms, "
              f"batch 128 = {cost[name]['lat_b128']:.3f} ms")
    return cost


@torch.no_grad()
def accuracy(model, loader):
    model.eval()
    correct = sum((model(x).argmax(1) == y).sum().item() for x, y in loader)
    return correct / len(loader.dataset)


def train(model, train_dl, val_dl, test_dl, name):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for x, y in train_dl:
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()
        print(f"[{name}] epoch {epoch}: val acc = {accuracy(model, val_dl):.4f}")
    test_acc = accuracy(model, test_dl)
    print(f"[{name}] test acc = {test_acc:.4f}")
    return test_acc


def scatter(ax, xs, acc, xlabel, unit):
    for name, marker in zip(acc, ["o", "s", "^"]):
        ax.scatter(xs[name], acc[name], s=80, marker=marker, label=name)
        ax.annotate(f"{name}\n{xs[name]:,.3g} {unit}", (xs[name], 100 * acc[name]),
                    textcoords="offset points", xytext=(8, -4), fontsize=8, va="top")
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.margins(x=0.5)
    ax.grid(alpha=0.3)


def make_plots(cost, acc):
    acc_pct = {n: 100 * a for n, a in acc.items()}

    fig, ax = plt.subplots(figsize=(6, 4))
    for name, marker in zip(acc, ["o", "s", "^"]):
        ax.scatter(cost[name]["params"], acc_pct[name], s=80, marker=marker, label=name)
    ax.set_xscale("log")
    ax.set_xlabel("Parameters (log scale)")
    ax.set_ylabel("Test accuracy, %")
    ax.set_title("Accuracy vs parameters")
    ax.margins(x=0.3, y=0.3)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "acc_vs_params.png", dpi=150)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, key, title in [(axes[0], "lat_b1", "Batch 1"), (axes[1], "lat_b128", "Batch 128")]:
        for name, marker in zip(acc, ["o", "s", "^"]):
            ax.scatter(cost[name][key], acc_pct[name], s=80, marker=marker, label=name)
        ax.set_xscale("log")
        ax.set_xlabel("Latency, ms (log scale)")
        ax.set_title(title)
        ax.margins(x=0.3, y=0.3)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Test accuracy, %")
    axes[0].legend()
    fig.suptitle("Accuracy vs inference latency (CPU)")
    fig.tight_layout()
    fig.savefig(OUT / "acc_vs_latency.png", dpi=150)


def main():
    set_seed()
    torch.set_num_threads(2)
    print_versions()

    models = build_models()
    cost = measure_cost(models)

    train_dl, val_dl, test_dl = get_loaders()
    acc = {}
    for name in models:
        set_seed()
        model = build_models()[name]
        acc[name] = train(model, train_dl, val_dl, test_dl, name)

    make_plots(cost, acc)
    print(f"\nPlots saved to {OUT}")


if __name__ == "__main__":
    main()