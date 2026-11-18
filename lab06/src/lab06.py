import argparse
import copy
import os
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

try:
    import resource
except ImportError:
    resource = None


# ===== Task 1: YAML config =====
def set_by_path(cfg, dotted_key, value):
    node = cfg
    keys = dotted_key.split(".")
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def load_config(path, overrides=None):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for item in overrides or []:
        key, value = item.split("=", 1)
        set_by_path(cfg, key, yaml.safe_load(value))
    return cfg


def with_overrides(cfg, **dotted):
    new = copy.deepcopy(cfg)
    for k, v in dotted.items():
        set_by_path(new, k, v)
    return new


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flatten(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


class Lab3CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10, dropout=0.3):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(block(in_channels, 32), block(32, 64), block(64, 128))
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                        nn.Dropout(dropout), nn.Linear(128, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(cfg):
    m = cfg["model"]
    if m["name"] == "Lab3CNN":
        return Lab3CNN(m.get("in_channels", 1), m["num_classes"], m.get("dropout", 0.0))
    raise ValueError(f"Unknown model {m['name']}")


# ===== Task 1: data =====
def _seed_worker(worker_id):
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def get_loaders(cfg, generator=None):
    d = cfg["data"]
    if d["name"] == "fashionmnist":
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
        train = datasets.FashionMNIST(d["root"], train=True, download=True, transform=tf)
        test = datasets.FashionMNIST(d["root"], train=False, download=True, transform=tf)
    elif d["name"] == "cifar10":
        norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        train_tf = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(), norm])
        test_tf = transforms.Compose([transforms.ToTensor(), norm])
        train = datasets.CIFAR10(d["root"], train=True, download=True, transform=train_tf)
        test = datasets.CIFAR10(d["root"], train=False, download=True, transform=test_tf)
    elif d["name"] == "fake":
        shape = (cfg["model"].get("in_channels", 1), 28, 28)
        train = datasets.FakeData(5000, shape, 10, transform=transforms.ToTensor(), random_offset=0)
        test = datasets.FakeData(1000, shape, 10, transform=transforms.ToTensor(), random_offset=10_000)
    else:
        raise ValueError(d["name"])
    if d.get("subset"):
        n = int(d["subset"])
        train = Subset(train, range(min(n, len(train))))
        test = Subset(test, range(min(n // 5, len(test))))

    pin = d.get("pin_memory", False) and torch.cuda.is_available()
    common = dict(num_workers=d["num_workers"], pin_memory=pin, worker_init_fn=_seed_worker,
                  persistent_workers=d["num_workers"] > 0)
    train_loader = DataLoader(train, batch_size=d["batch_size"], shuffle=True, generator=generator, **common)
    test_loader = DataLoader(test, batch_size=512, shuffle=False, **common)
    return train_loader, test_loader


# ===== Task 1 =====
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def reset_peak_memory(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()


def peak_memory_mb(device):
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 2**20
    if resource is not None:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return rss / 2**20 if os.uname().sysname == "Darwin" else rss / 1024
    import psutil
    return psutil.Process().memory_info().peak_wset / 2**20


def build_optimizer(cfg, params):
    t = cfg["train"]
    if t["optimizer"] == "sgd":
        return torch.optim.SGD(params, lr=t["lr"], momentum=t["momentum"], weight_decay=t["weight_decay"])
    if t["optimizer"] == "adam":
        return torch.optim.Adam(params, lr=t["lr"], weight_decay=t["weight_decay"])
    raise ValueError(t["optimizer"])


def build_scheduler(cfg, opt):
    t = cfg["train"]
    s = t["scheduler"]
    if s in (None, "none"):
        return None
    if s == "step":
        return torch.optim.lr_scheduler.StepLR(opt, step_size=t["step_size"], gamma=t["gamma"])
    if s == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t["epochs"])
    if s == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=t["gamma"],
                                                          patience=t["plateau_patience"])
    raise ValueError(s)


# ===== Task 1 + Task 6 =====
def train_one_epoch(model, loader, opt, criterion, device, accum_steps=1):
    model.train()
    total_loss, correct, seen = 0.0, 0, 0
    opt.zero_grad(set_to_none=True)
    n_batches = len(loader)
    sync(device)
    t0 = time.perf_counter()
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(x)
        loss = criterion(out, y)
        (loss / accum_steps).backward()  # Task 6
        if (i + 1) % accum_steps == 0 or (i + 1) == n_batches:
            opt.step()
            opt.zero_grad(set_to_none=True)
        total_loss += loss.item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        seen += y.size(0)
    sync(device)
    elapsed = time.perf_counter() - t0
    return total_loss / seen, correct / seen, seen / elapsed, elapsed


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, seen = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        total_loss += criterion(out, y).item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        seen += y.size(0)
    return total_loss / seen, correct / seen


# ===== Task 1 + Task 7 =====
def save_checkpoint(path, **state):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path + ".tmp")
    os.replace(path + ".tmp", path)


def rng_state(generator):
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "loader": generator.get_state()}


def load_rng_state(state, generator):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    generator.set_state(state["loader"])


# ===== Task 1 =====
def run(cfg):
    set_seed(cfg["seed"])
    device = get_device(cfg.get("device", "auto"))
    g = torch.Generator().manual_seed(cfg["seed"])
    train_loader, test_loader = get_loaders(cfg, generator=g)

    model = build_model(cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    opt = build_optimizer(cfg, model.parameters())
    sched = build_scheduler(cfg, opt)
    t = cfg["train"]
    ckpt_dir = os.path.join(cfg["checkpoint"]["dir"], cfg["run_name"])

    start_epoch, history, best_acc, run_id = 0, [], 0.0, None
    if cfg["checkpoint"].get("resume"):  # Task 7
        ck = torch.load(cfg["checkpoint"]["resume"], map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        if sched is not None and ck["scheduler"] is not None:
            sched.load_state_dict(ck["scheduler"])
        load_rng_state(ck["rng"], g)
        start_epoch, history, best_acc, run_id = ck["epoch"], ck["history"], ck["best_acc"], ck["mlflow_run_id"]
        print(f"Resumed from {cfg['checkpoint']['resume']}: epoch {start_epoch}, best_acc={best_acc:.4f}")

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["experiment"])
    with mlflow.start_run(run_id=run_id, run_name=None if run_id else cfg["run_name"]) as mrun:
        if run_id is None:
            mlflow.log_params(flatten(cfg))
            mlflow.log_param("resolved_device", str(device))
            mlflow.log_param("n_params", sum(p.numel() for p in model.parameters()))

        for epoch in range(start_epoch, t["epochs"]):
            reset_peak_memory(device)
            tr_loss, tr_acc, sps, ep_time = train_one_epoch(model, train_loader, opt, criterion, device,
                                                            t.get("accum_steps", 1))
            va_loss, va_acc = evaluate(model, test_loader, criterion, device)
            lr = opt.param_groups[0]["lr"]
            if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                sched.step(va_loss)
            elif sched is not None:
                sched.step()

            row = dict(epoch=epoch + 1, lr=lr, train_loss=tr_loss, train_acc=tr_acc, val_loss=va_loss,
                       val_acc=va_acc, samples_per_s=sps, epoch_time_s=ep_time,
                       peak_mem_mb=peak_memory_mb(device))
            history.append(row)
            mlflow.log_metrics({k: v for k, v in row.items() if k != "epoch"}, step=epoch + 1)
            print(" | ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))

            state = dict(model=model.state_dict(), optimizer=opt.state_dict(),
                         scheduler=sched.state_dict() if sched else None, epoch=epoch + 1,
                         history=history, best_acc=max(best_acc, va_acc), rng=rng_state(g),
                         mlflow_run_id=mrun.info.run_id, config=cfg)
            save_checkpoint(os.path.join(ckpt_dir, "last.pt"), **state)
            save_checkpoint(os.path.join(ckpt_dir, f"epoch_{epoch + 1:03d}.pt"), **state)
            if va_acc > best_acc:
                best_acc = va_acc
                save_checkpoint(os.path.join(ckpt_dir, "best.pt"), **state)

            if t.get("stop_after_epoch") and epoch + 1 == t["stop_after_epoch"]:  # Task 7
                print(f"*** Training interrupted after epoch {epoch + 1} (simulated) ***")
                break

        os.makedirs(ckpt_dir, exist_ok=True)
        cfg_path = os.path.join(ckpt_dir, "config.yaml")
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        hist_path = os.path.join(ckpt_dir, "history.csv")
        pd.DataFrame(history).to_csv(hist_path, index=False)
        mlflow.log_artifact(cfg_path)
        mlflow.log_artifact(hist_path)
        if os.path.exists(os.path.join(ckpt_dir, "best.pt")):
            mlflow.log_artifact(os.path.join(ckpt_dir, "best.pt"), artifact_path="checkpoints")
        mlflow.log_metric("best_val_acc", best_acc)
    return history


def cmd_train(cfg):
    run(cfg)


# ===== Task 2 =====
def cmd_profile(cfg):
    out = "results/profile"
    os.makedirs(out, exist_ok=True)
    phases = ["data_loading", "forward", "backward", "optimizer_step"]
    set_seed(cfg["seed"])
    device = get_device(cfg.get("device", "auto"))
    train_loader, _ = get_loaders(cfg, torch.Generator().manual_seed(cfg["seed"]))
    model = build_model(cfg).to(device)
    opt = build_optimizer(cfg, model.parameters())
    crit = nn.CrossEntropyLoss()
    model.train()

    activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device.type == "cuda" else [])
    with profile(activities=activities, record_shapes=False, profile_memory=False) as prof:
        it = iter(train_loader)
        while True:
            with record_function("data_loading"):
                try:
                    x, y = next(it)
                except StopIteration:
                    break
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                sync(device)
            with record_function("forward"):
                loss = crit(model(x), y)
                sync(device)
            with record_function("backward"):
                opt.zero_grad(set_to_none=True)
                loss.backward()
                sync(device)
            with record_function("optimizer_step"):
                opt.step()
                sync(device)

    events = {e.key: e for e in prof.key_averages()}
    df = pd.DataFrame([{"phase": ph, "time_ms": events[ph].cpu_time_total / 1000 if ph in events else 0.0}
                       for ph in phases])
    df["share_%"] = 100 * df.time_ms / df.time_ms.sum()
    print(df.to_string(index=False, float_format="%.2f"))
    df.to_csv(f"{out}/profile_breakdown.csv", index=False)
    sort_key = "cuda_time_total" if device.type == "cuda" else "cpu_time_total"
    with open(f"{out}/profiler_top_ops.txt", "w") as f:
        f.write(prof.key_averages().table(sort_by=sort_key, row_limit=25))
    prof.export_chrome_trace(f"{out}/trace.json")

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["experiment"])
    with mlflow.start_run(run_name="profile_one_epoch"):
        mlflow.log_params({"num_workers": cfg["data"]["num_workers"],
                           "batch_size": cfg["data"]["batch_size"], "device": str(device)})
        for _, r in df.iterrows():
            mlflow.log_metric(f"time_ms_{r.phase}", r.time_ms)
            mlflow.log_metric(f"share_{r.phase}", r["share_%"])
        mlflow.log_artifact(f"{out}/profile_breakdown.csv")
        mlflow.log_artifact(f"{out}/profiler_top_ops.txt")


# ===== Task 3 =====
def cmd_workers(cfg_base):
    out = "results/workers"
    os.makedirs(out, exist_ok=True)
    workers = [0, 1, 2, 4, 8]
    rows = []
    for w in workers:
        cfg = with_overrides(cfg_base, **{"data.num_workers": w, "train.epochs": 1, "run_name": f"workers_{w}"})
        loader, _ = get_loaders(cfg, torch.Generator().manual_seed(cfg["seed"]))
        n, t0 = 0, time.perf_counter()
        for x, _ in loader:
            n += x.size(0)
        loader_sps = n / (time.perf_counter() - t0)
        h = run(cfg)[-1]
        rows.append({"num_workers": w, "loader_only_samples_s": loader_sps,
                     "train_samples_s": h["samples_per_s"], "epoch_time_s": h["epoch_time_s"]})

    df = pd.DataFrame(rows)
    df.to_csv(f"{out}/workers.csv", index=False)
    print(df.to_string(index=False, float_format="%.1f"))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df.num_workers, df.loader_only_samples_s, "o-", label="DataLoader only")
    ax.plot(df.num_workers, df.train_samples_s, "s-", label="Full training step")
    ax.set(xlabel="num_workers", ylabel="throughput, samples/s", title="Throughput vs num_workers",
           xticks=workers)
    ax.grid(alpha=.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{out}/workers.png", dpi=150)


# ===== Task 4 =====
def cmd_batch(cfg_base):
    out = "results/batch"
    os.makedirs(out, exist_ok=True)
    batches = [32, 64, 128, 256, 512]
    base_lr = cfg_base["train"]["lr"]
    rows = []
    for bs in batches:
        cfg = with_overrides(cfg_base, **{"data.batch_size": bs, "train.epochs": 5,
                                          "train.lr": base_lr * bs / 128, "run_name": f"bs_{bs}"})
        try:
            h = run(cfg)
            rows.append({"batch_size": bs, "lr": cfg["train"]["lr"],
                         "throughput_samples_s": np.mean([r["samples_per_s"] for r in h]),
                         "peak_mem_mb": max(r["peak_mem_mb"] for r in h), "final_val_acc": h[-1]["val_acc"]})
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            rows.append({"batch_size": bs, "lr": cfg["train"]["lr"], "throughput_samples_s": np.nan,
                         "peak_mem_mb": np.nan, "final_val_acc": np.nan})
            print(f"batch {bs}: OOM")

    df = pd.DataFrame(rows)
    df.to_csv(f"{out}/batch_study.csv", index=False)
    print(df.to_string(index=False, float_format="%.3f"))

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, col, title in zip(axes, ["throughput_samples_s", "peak_mem_mb", "final_val_acc"],
                              ["Throughput, samples/s", "Peak memory, MB", "Val accuracy after 5 ep"]):
        ax.plot(df.batch_size, df[col], "o-")
        ax.set_xscale("log", base=2)
        ax.set_xticks(batches)
        ax.set_xticklabels(batches)
        ax.set(xlabel="batch size", title=title)
        ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(f"{out}/batch_study.png", dpi=150)


# ===== Task 5=====
def cmd_sched(cfg_base):
    out = "results/sched_opt"
    os.makedirs(out, exist_ok=True)
    optimizers = {"sgd": 0.05, "adam": 1e-3}
    schedulers = ["step", "cosine", "plateau"]
    rows, curves = [], {}
    for opt_name, lr in optimizers.items():
        for sch in schedulers:
            name = f"{opt_name}_{sch}"
            cfg = with_overrides(cfg_base, **{"train.optimizer": opt_name, "train.lr": lr,
                                              "train.scheduler": sch, "train.epochs": 10, "run_name": name})
            h = pd.DataFrame(run(cfg))
            curves[name] = h
            rows.append({"optimizer": opt_name, "scheduler": sch, "lr0": lr,
                         "final_val_acc": h.val_acc.iloc[-1], "best_val_acc": h.val_acc.max(),
                         "final_train_loss": h.train_loss.iloc[-1], "total_time_s": h.epoch_time_s.sum()})

    df = pd.DataFrame(rows)
    df.to_csv(f"{out}/sched_opt.csv", index=False)
    print(df.to_string(index=False, float_format="%.4f"))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for name, h in curves.items():
        ls = "-" if name.startswith("sgd") else "--"
        axes[0].plot(h.epoch, h.val_acc, ls, marker="o", ms=3, label=name)
        axes[1].plot(h.epoch, h.train_loss, ls, marker="o", ms=3, label=name)
        axes[2].plot(h.epoch, h.lr / h.lr.iloc[0], ls, marker="o", ms=3, label=name)
    for ax, title in zip(axes, ["Val accuracy", "Train loss", "LR / LR0"]):
        ax.set(xlabel="epoch", title=title)
        ax.grid(alpha=.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{out}/sched_opt.png", dpi=150)


# ===== Task 6 =====
def cmd_accum(cfg_base):
    out = "results/accum"
    os.makedirs(out, exist_ok=True)
    setups = {"bs256_accum1": {"data.batch_size": 256, "train.accum_steps": 1},
              "bs64_accum4": {"data.batch_size": 64, "train.accum_steps": 4}}
    curves, rows = {}, []
    for name, ov in setups.items():
        h = pd.DataFrame(run(with_overrides(cfg_base, **ov, run_name=name)))
        curves[name] = h
        rows.append({"setup": name, "final_train_loss": h.train_loss.iloc[-1], "final_val_acc": h.val_acc.iloc[-1],
                     "samples_per_s": h.samples_per_s.mean(), "peak_mem_mb": h.peak_mem_mb.max()})

    df = pd.DataFrame(rows)
    df.to_csv(f"{out}/accum.csv", index=False)
    a, b = curves["bs256_accum1"], curves["bs64_accum4"]
    print(df.to_string(index=False, float_format="%.4f"))
    print(f"max |Δ train_loss| = {(a.train_loss - b.train_loss).abs().max():.4f}")
    print(f"max |Δ val_acc|    = {(a.val_acc - b.val_acc).abs().max():.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, h in curves.items():
        axes[0].plot(h.epoch, h.train_loss, "o-", label=name)
        axes[1].plot(h.epoch, h.val_acc, "o-", label=name)
    for ax, title in zip(axes, ["Train loss", "Val accuracy"]):
        ax.set(xlabel="epoch", title=title)
        ax.grid(alpha=.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(f"{out}/accum.png", dpi=150)


# ===== Task 7 =====
def cmd_resume(cfg_base):
    out = "results/resume"
    os.makedirs(out, exist_ok=True)
    epochs = cfg_base["train"]["epochs"]
    ck_dir = cfg_base["checkpoint"]["dir"]

    print("=== 1) reference run without interruption ===")
    full = pd.DataFrame(run(with_overrides(cfg_base, run_name="no_interrupt")))
    print("=== 2) training is interrupted after epoch 3 ===")
    run(with_overrides(cfg_base, run_name="resume_demo", **{"train.stop_after_epoch": 3}))
    print("=== 3) resuming from checkpoint ===")
    resumed = pd.DataFrame(run(with_overrides(cfg_base, run_name="resume_demo", **{
        "checkpoint.resume": os.path.join(ck_dir, "resume_demo", "last.pt")})))

    cmp = full[["epoch", "train_loss", "val_acc"]].merge(
        resumed[["epoch", "train_loss", "val_acc"]], on="epoch", suffixes=("_full", "_resumed"))
    cmp["abs_diff_val_acc"] = (cmp.val_acc_full - cmp.val_acc_resumed).abs()
    cmp.to_csv(f"{out}/resume_vs_full.csv", index=False)
    print(cmp.to_string(index=False, float_format="%.4f"))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, col, title in zip(axes, ["train_loss", "val_acc"], ["Train loss", "Val accuracy"]):
        ax.plot(full.epoch, full[col], "o-", label="uninterrupted", alpha=.6)
        ax.plot(resumed.epoch, resumed[col], "x--", label="interrupted at 3 + resumed")
        ax.axvline(3.5, color="red", ls=":", label="resume point")
        ax.set(xlabel="epoch", title=title, xticks=range(1, epochs + 1))
        ax.grid(alpha=.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(f"{out}/resume.png", dpi=150)

# ===== MLflow runs export =====
def cmd_export(cfg):
    os.makedirs("results", exist_ok=True)
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    runs = mlflow.search_runs(experiment_names=[cfg["experiment"]])
    cols = [c for c in runs.columns if c.startswith(("tags.mlflow.runName", "params.train.", "params.data.batch_size",
                                                     "params.data.num_workers", "metrics.")) or c == "run_id"]
    runs[cols].to_csv("results/mlflow_runs.csv", index=False)
    print(f"{len(runs)} runs -> results/mlflow_runs.csv")


# ===== CLI =====
COMMANDS = {"train": cmd_train, "profile": cmd_profile, "workers": cmd_workers, "batch": cmd_batch,
            "sched": cmd_sched, "accum": cmd_accum, "resume": cmd_resume, "export": cmd_export}

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Lab 6 training pipeline")
    p.add_argument("command", choices=COMMANDS)
    p.add_argument("--config", default="src/config.yaml")
    p.add_argument("--set", nargs="*", default=[])
    args = p.parse_args()
    COMMANDS[args.command](load_config(args.config, args.set))