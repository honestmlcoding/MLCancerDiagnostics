#!/usr/bin/env python3

import argparse
import csv
import os
import random
import socket
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler


# ---------------------------
# Seeding
# ---------------------------

def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------
# DDP helpers
# ---------------------------

def ddp_is_enabled() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()

def ddp_rank() -> int:
    return torch.distributed.get_rank() if ddp_is_enabled() else 0

def ddp_world_size() -> int:
    return torch.distributed.get_world_size() if ddp_is_enabled() else 1

def ddp_barrier():
    if ddp_is_enabled():
        torch.distributed.barrier()

def log_rank0(msg: str):
    if ddp_rank() == 0:
        print(msg, flush=True)

@torch.no_grad()
def all_reduce_sum(x: torch.Tensor) -> torch.Tensor:
    if not ddp_is_enabled():
        return x
    y = x.clone()
    torch.distributed.all_reduce(y, op=torch.distributed.ReduceOp.SUM)
    return y

@torch.no_grad()
def all_reduce_mean(x: torch.Tensor) -> torch.Tensor:
    if not ddp_is_enabled():
        return x
    y = x.clone()
    torch.distributed.all_reduce(y, op=torch.distributed.ReduceOp.SUM)
    y /= ddp_world_size()
    return y

@torch.no_grad()
def ddp_all_gather_cat_1d(x: torch.Tensor) -> torch.Tensor:
    if not ddp_is_enabled():
        return x
    if x.dim() != 1:
        raise ValueError("ddp_all_gather_cat_1d expects a 1D tensor.")

    device = x.device
    n_local = torch.tensor([x.numel()], device=device, dtype=torch.long)
    ns_t = [torch.zeros_like(n_local) for _ in range(ddp_world_size())]
    torch.distributed.all_gather(ns_t, n_local)
    # ns_t is a list of 1-element tensors; convert to python ints
    ns = [int(t.item()) for t in ns_t]
    max_n = int(max(ns))

    if x.numel() < max_n:
        x_pad = torch.cat([x, torch.zeros(max_n - x.numel(), device=device, dtype=x.dtype)], dim=0)
    else:
        x_pad = x

    gathered = [torch.zeros(max_n, device=device, dtype=x.dtype) for _ in range(ddp_world_size())]
    torch.distributed.all_gather(gathered, x_pad)

    chunks = [g[:int(n)] for g, n in zip(gathered, ns)]
    return torch.cat(chunks, dim=0)


def _set_default_ddp_env():
    os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("NCCL_DEBUG", "WARN")

def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


# ---------------------------
# CSV loading
# ---------------------------

def _sniff_dialect(path: str) -> csv.Dialect:
    with open(path, "r", encoding="utf-8") as f:
        sample = f.read(8192)
        f.seek(0)
        try:
            return csv.Sniffer().sniff(sample, delimiters=",\t;")
        except csv.Error:
            class _D(csv.Dialect):
                delimiter = ","
                quotechar = '"'
                escapechar = None
                doublequote = True
                skipinitialspace = False
                lineterminator = "\n"
                quoting = csv.QUOTE_MINIMAL
            return _D()

def _norm_label(v: str) -> str:
    return str(v).strip().lower()

def label_to_binary(v, positive_label: str) -> int:
    s = _norm_label(v)
    pos = _norm_label(positive_label)

    if s == pos:
        return 1

    pos_aliases = {"m", "malignant", "cancer", "positive", "1", "true", "yes"}
    neg_aliases = {"b", "benign", "negative", "0", "false", "no"}

    if pos in pos_aliases and s in pos_aliases:
        return 1
    if s in neg_aliases:
        return 0
    if s in pos_aliases:
        return 1

    try:
        f = float(s)
        return 1 if f > 0.5 else 0
    except Exception as e:
        raise ValueError(f"Unrecognized label '{v}'. Use --positive_label.") from e

def load_tabular_csv(
    path: str,
    label_col: str,
    positive_label: str,
    drop_cols: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    drop_cols = drop_cols or []
    dialect = _sniff_dialect(path)

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, dialect=dialect)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header row.")
        cols = [c.strip() for c in reader.fieldnames]

        if label_col not in cols:
            raise ValueError(f"Label column '{label_col}' not found. Available columns: {cols}")

        candidate = [c for c in cols if c != label_col and c not in set(drop_cols)]
        rows: List[Dict[str, str]] = [r for r in reader if r is not None]

    if len(rows) == 0:
        raise ValueError("No data rows found in CSV.")

    numeric_cols: List[str] = []
    dropped_non_numeric: List[str] = []

    for c in candidate:
        ok = True
        for r in rows:
            v = r.get(c, "")
            if v is None:
                continue
            v = str(v).strip()
            if v == "" or v.lower() == "nan":
                continue
            try:
                float(v)
            except Exception:
                ok = False
                break
        if ok:
            numeric_cols.append(c)
        else:
            dropped_non_numeric.append(c)

    if len(numeric_cols) == 0:
        raise ValueError("No numeric feature columns found.")

    if dropped_non_numeric:
        log_rank0(f"Warning: dropping non-numeric feature columns: {dropped_non_numeric}")

    N = len(rows)
    D = len(numeric_cols)
    X = np.zeros((N, D), dtype=np.float32)
    y = np.zeros((N,), dtype=np.int64)

    for i, r in enumerate(rows):
        y[i] = label_to_binary(r.get(label_col, ""), positive_label=positive_label)
        for j, c in enumerate(numeric_cols):
            v = r.get(c, "")
            v = "nan" if v is None else str(v).strip()
            try:
                X[i, j] = float(v)
            except Exception:
                X[i, j] = np.nan

    with np.errstate(all="ignore"):
        col_means = np.nanmean(X, axis=0)
    col_means = np.where(np.isfinite(col_means), col_means, 0.0).astype(np.float32)

    inds = np.where(np.isnan(X))
    if inds[0].size > 0:
        X[inds] = col_means[inds[1]]

    return X, y, numeric_cols


# ---------------------------
# Dataset / Model
# ---------------------------

class TabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]

class MLPBinary(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        if num_layers < 1:
            raise ValueError("--num_layers must be >= 1")
        layers: List[nn.Module] = []
        d = in_dim
        for _ in range(num_layers):
            layers += [nn.Linear(d, hidden_dim), nn.GELU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = hidden_dim
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ---------------------------
# Metrics
# ---------------------------

@torch.no_grad()
def binary_auc_roc(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    scores = scores.float()
    labels = labels.float()
    n_pos = labels.sum()
    n_neg = (1.0 - labels).sum()
    if n_pos < 1 or n_neg < 1:
        return torch.tensor(float("nan"), device=scores.device)

    order = torch.argsort(scores)
    ranks = torch.empty_like(order, dtype=torch.float)
    ranks[order] = torch.arange(1, scores.numel() + 1, device=scores.device, dtype=torch.float)

    sorted_scores = scores[order]
    sorted_ranks = ranks[order]
    i = 0
    while i < sorted_scores.numel():
        j = i + 1
        while j < sorted_scores.numel() and sorted_scores[j] == sorted_scores[i]:
            j += 1
        if j - i > 1:
            sorted_ranks[i:j] = sorted_ranks[i:j].mean()
        i = j
    ranks[order] = sorted_ranks

    sum_ranks_pos = (ranks * labels).sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
    return auc.clamp(0.0, 1.0)

@torch.no_grad()
def eval_model(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool) -> Dict[str, float]:
    model.eval()
    tp = torch.tensor(0.0, device=device)
    fp = torch.tensor(0.0, device=device)
    tn = torch.tensor(0.0, device=device)
    fn = torch.tensor(0.0, device=device)
    loss_sum = torch.tensor(0.0, device=device)
    n_sum = torch.tensor(0.0, device=device)

    scores_local = []
    labels_local = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=(amp and device.type == "cuda")):
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y, reduction="sum")
        probs = torch.sigmoid(logits)
        pred = (probs >= 0.5).float()

        tp += (pred * y).sum()
        fp += (pred * (1.0 - y)).sum()
        tn += ((1.0 - pred) * (1.0 - y)).sum()
        fn += ((1.0 - pred) * y).sum()
        loss_sum += loss
        n_sum += y.numel()

        scores_local.append(probs.detach().flatten())
        labels_local.append(y.detach().flatten())

    tp = all_reduce_sum(tp); fp = all_reduce_sum(fp); tn = all_reduce_sum(tn); fn = all_reduce_sum(fn)
    loss_sum = all_reduce_sum(loss_sum); n_sum = all_reduce_sum(n_sum)

    eps = 1e-12
    acc = (tp + tn) / (tp + tn + fp + fn + eps)
    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    f1 = 2 * prec * rec / (prec + rec + eps)
    avg_loss = loss_sum / (n_sum + eps)

    scores = torch.cat(scores_local, dim=0) if scores_local else torch.empty(0, device=device)
    labels = torch.cat(labels_local, dim=0) if labels_local else torch.empty(0, device=device)
    scores_g = ddp_all_gather_cat_1d(scores)
    labels_g = ddp_all_gather_cat_1d(labels)
    auc = binary_auc_roc(scores_g, labels_g)

    return {
        "loss": float(avg_loss.item()),
        "acc": float(acc.item()),
        "precision": float(prec.item()),
        "recall": float(rec.item()),
        "f1": float(f1.item()),
        "auroc": float(auc.item()) if torch.isfinite(auc) else float("nan"),
        "tp": float(tp.item()),
        "fp": float(fp.item()),
        "tn": float(tn.item()),
        "fn": float(fn.item()),
    }


# ---------------------------
# Training
# ---------------------------

@dataclass
class Config:
    batch_size: int = 32
    epochs: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    hidden_dim: int = 16
    num_layers: int = 4
    dropout: float = 0.1
    amp: bool = True
    log_every: int = 50
    eval_every: int = 200
    num_workers: int = 0


def _run_worker(local_rank: int, args: argparse.Namespace):
    _set_default_ddp_env()
    using_ddp = ("RANK" in os.environ and "WORLD_SIZE" in os.environ)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda")
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    if using_ddp and (not ddp_is_enabled()):
        torch.distributed.init_process_group(backend=backend)

    seed_all(args.seed + ddp_rank())

    cfg = Config(
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        amp=not args.no_amp,
        log_every=args.log_every,
        eval_every=args.eval_every,
        num_workers=args.num_workers,
    )

    drop_cols = [c.strip() for c in args.drop_cols.split(",") if c.strip()]
    X, y, _feat_names = load_tabular_csv(args.data_path, args.label_col, args.positive_label, drop_cols)

    if ddp_rank() == 0:
        log_rank0(f"Loaded {X.shape[0]} rows, {X.shape[1]} numeric features from {args.data_path}")
        log_rank0(f"Positive='{args.positive_label}' mapped to 1. Pos rate={float(y.mean()):.3f}")

    rng = np.random.RandomState(args.seed)
    idx = np.arange(X.shape[0]); rng.shuffle(idx)
    X = X[idx]; y = y[idx]

    n_train = int(X.shape[0] * args.train_split)
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:], y[n_train:]

    mu = X_train.mean(axis=0, keepdims=True)
    sigma = X_train.std(axis=0, keepdims=True)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    X_train = (X_train - mu) / sigma
    X_val = (X_val - mu) / sigma

    train_ds = TabularDataset(X_train, y_train)
    val_ds = TabularDataset(X_val, y_val)

    train_sampler = DistributedSampler(train_ds, shuffle=True) if ddp_is_enabled() else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if ddp_is_enabled() else None

    pin_memory = (device.type == "cuda")
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, sampler=train_sampler,
        shuffle=(train_sampler is None), num_workers=cfg.num_workers,
        pin_memory=pin_memory, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=min(cfg.batch_size, 1024), sampler=val_sampler,
        shuffle=False, num_workers=cfg.num_workers, pin_memory=pin_memory, drop_last=False
    )

    model = MLPBinary(X.shape[1], cfg.hidden_dim, cfg.num_layers, cfg.dropout).to(device)

    if ddp_is_enabled():
        if device.type == "cuda":
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
        else:
            model = torch.nn.parallel.DistributedDataParallel(model)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))

    pos_rate = float(y_train.mean()) if len(y_train) else 0.0
    log_rank0(f"Baseline majority-class accuracy (train split): {max(pos_rate, 1.0-pos_rate):.4f}")

    global_step = 0
    t0 = time.time()

    for epoch in range(cfg.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        for xb, yb in train_loader:
            global_step += 1
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(cfg.amp and device.type == "cuda")):
                logits = model(xb)
                loss = F.binary_cross_entropy_with_logits(logits, yb)

            scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)

            if global_step % cfg.log_every == 0:
                loss_r = all_reduce_mean(loss.detach())
                elapsed = time.time() - t0
                examples = cfg.batch_size * cfg.log_every * ddp_world_size()
                ex_s = examples / max(elapsed, 1e-6)
                t0 = time.time()
                log_rank0(f"[epoch {epoch+1}/{cfg.epochs} step {global_step}] train_loss={loss_r.item():.4f} ex/s={ex_s:,.0f}")

            if global_step % cfg.eval_every == 0:
                ddp_barrier()
                m = eval_model(model, val_loader, device=device, amp=cfg.amp)
                log_rank0(f"  eval: loss={m['loss']:.4f} acc={m['acc']:.4f} prec={m['precision']:.4f} rec={m['recall']:.4f} f1={m['f1']:.4f} auroc={m['auroc']:.4f}")
                ddp_barrier()

    ddp_barrier()
    m = eval_model(model, val_loader, device=device, amp=cfg.amp)
    log_rank0(f"FINAL eval: loss={m['loss']:.4f} acc={m['acc']:.4f} prec={m['precision']:.4f} rec={m['recall']:.4f} f1={m['f1']:.4f} auroc={m['auroc']:.4f} (tp={int(m['tp'])} fp={int(m['fp'])} tn={int(m['tn'])} fn={int(m['fn'])})")

    if ddp_is_enabled():
        torch.distributed.destroy_process_group()


# TOP-LEVEL entrypoint for multiprocessing.spawn (must be pickleable)
def spawn_entry(local_rank: int, args: argparse.Namespace):
    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    _run_worker(local_rank, args)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--label_col", type=str, default="diagnosis")
    p.add_argument("--positive_label", type=str, default="M")
    p.add_argument("--drop_cols", type=str, default="id,Unnamed: 32")
    p.add_argument("--train_split", type=float, default=0.85)
    p.add_argument("--seed", type=int, default=1234)

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--hidden_dim", type=int, default=16)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--no_amp", action="store_true")

    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--eval_every", type=int, default=200)

    p.add_argument("--gpus", type=int, default=2, help="GPUs to use when launching with plain python (default: 2)")

    args = p.parse_args()

    launched_by_torchrun = ("LOCAL_RANK" in os.environ and "RANK" in os.environ and "WORLD_SIZE" in os.environ)
    if launched_by_torchrun:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        _run_worker(local_rank, args)
        return

    req = int(args.gpus)
    avail = torch.cuda.device_count()
    if req > 1:
        if avail < req:
            raise RuntimeError(f"Requested --gpus {req} but only {avail} CUDA devices visible.")

        _set_default_ddp_env()
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(_find_free_port())
        os.environ["WORLD_SIZE"] = str(req)

        torch.multiprocessing.spawn(spawn_entry, args=(args,), nprocs=req, join=True)
    else:
        _run_worker(0, args)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback, sys
        print("\nFATAL ERROR:", file=sys.stderr)
        traceback.print_exc()
        raise
