import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import gc
import hashlib
import numpy as np
import pandas as pd
import polars as pl
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from torch.cuda.amp import autocast, GradScaler

from trendyol_metric_group_auc import score as group_auc_score


def _print(msg: str):
    print(f"[wide_deep] {msg}")


# -------------------------
# Feature utils
# -------------------------

def build_features(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    targets = ["ordered", "clicked"]
    exclude = set(targets + [
        "ts_hour",
        "session_id",
        "content_id_hashed",
        "content_creation_date",
        "update_date",
        "added_to_cart",
        "added_to_fav",
    ])
    # drop any datetime-like and typical time-named fields
    time_name_keywords = ("ts_", "_ts", "date", "time", "hour", "day", "week", "month", "year")
    for c in df.columns:
        dt = df[c].dtype
        if str(dt).startswith("datetime"):
            exclude.add(c)
        else:
            lc = c.lower()
            if any(k in lc for k in time_name_keywords):
                exclude.add(c)

    cols = [c for c in df.columns if c not in exclude]
    cat_cols, num_cols = [], []
    for c in cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            num_cols.append(c)
        else:
            cat_cols.append(c)
    return cat_cols, num_cols


def time_based_split(df_pl: pl.DataFrame, val_frac: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if "ts_hour" in df_pl.columns and df_pl["ts_hour"].dtype != pl.Datetime:
        df_pl = df_pl.with_columns(pl.col("ts_hour").str.to_datetime())
    df_pl = df_pl.sort("ts_hour")
    split_idx = int(len(df_pl) * (1 - val_frac))
    return df_pl[:split_idx].to_pandas(), df_pl[split_idx:].to_pandas()


# -------------------------
# Hashing helpers
# -------------------------

def stable_hash(s: str) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)


# -------------------------
# Dataset
# -------------------------

class WideDeepDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cat_cols: List[str], num_cols: List[str],
                 w_order: float, w_click: float,
                 num_means: Dict[str, float] | None = None,
                 num_stds: Dict[str, float] | None = None,
                 cat_bucket_sizes: Dict[str, int] | None = None,
                 wide_hash_dim: int = 200000):
        self.df = df.reset_index(drop=True)
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.w_order = w_order
        self.w_click = w_click
        self.num_means = num_means or {c: float(self.df[c].mean()) for c in num_cols}
        self.num_stds = num_stds or {c: float(self.df[c].std(ddof=0) or 1.0) for c in num_cols}
        self.cat_bucket_sizes = cat_bucket_sizes or {c: 50000 for c in cat_cols}
        self.wide_hash_dim = wide_hash_dim

        # Precompute numeric as np arrays
        self.num_arr = None
        if num_cols:
            arr = []
            for c in num_cols:
                mean, std = self.num_means[c], self.num_stds[c] or 1.0
                x = self.df[c].astype(float).fillna(mean).to_numpy()
                arr.append((x - mean) / (std if std != 0 else 1.0))
            self.num_arr = np.stack(arr, axis=1).astype(np.float32)

        # Labels
        y = ((self.df["ordered"] == 1) | (self.df["clicked"] == 1)).astype(np.float32).to_numpy()
        self.y = y

        # Sample weights
        rel = (self.w_order * self.df["ordered"].astype(float) + self.w_click * self.df["clicked"].astype(float)).to_numpy()
        pos = max(1.0, float(self.df["ordered"].sum() + self.df["clicked"].sum() > 0))
        # Use base weight similar to scale_pos_weight by prevalence
        pos_count = max(1.0, float(self.y.sum()))
        neg_count = max(1.0, float((self.y == 0).sum()))
        base = neg_count / pos_count
        self.w = np.where(self.y == 1.0, base * rel, 1.0).astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        # Wide indices (hash of feature=value)
        wide_idx = []
        for c in self.cat_cols:
            v = str(row[c])
            h = stable_hash(f"{c}={v}") % self.wide_hash_dim
            wide_idx.append(h)
        wide_idx = np.array(wide_idx, dtype=np.int64) if self.cat_cols else np.zeros((0,), dtype=np.int64)

        # Deep categorical hashed per feature
        deep_cat_idx = []
        for c in self.cat_cols:
            v = str(row[c])
            bucket = self.cat_bucket_sizes[c]
            h = stable_hash(f"{c}={v}") % bucket
            deep_cat_idx.append(h)
        deep_cat_idx = np.array(deep_cat_idx, dtype=np.int64) if self.cat_cols else np.zeros((0,), dtype=np.int64)

        # Numeric
        num_x = self.num_arr[idx] if self.num_arr is not None else np.zeros((0,), dtype=np.float32)

        y = self.y[idx]
        w = self.w[idx]
        return wide_idx, deep_cat_idx, num_x, y, w


# -------------------------
# Model
# -------------------------

class WideAndDeep(nn.Module):
    def __init__(self, cat_cols: List[str], num_dim: int,
                 cat_bucket_sizes: Dict[str, int],
                 wide_hash_dim: int = 200000,
                 emb_dim: int = 16,
                 hidden_dims: List[int] = [256, 128, 64],
                 dropout: float = 0.1):
        super().__init__()
        self.cat_cols = cat_cols
        self.num_dim = num_dim
        self.wide_hash_dim = wide_hash_dim

        # Wide: single embedding with dim=1 (linear weights), sum over features
        self.wide_emb = nn.Embedding(wide_hash_dim, 1)

        # Deep: per-categorical embeddings
        self.cat_embeddings = nn.ModuleDict()
        for c in cat_cols:
            self.cat_embeddings[c] = nn.Embedding(cat_bucket_sizes[c], emb_dim)

        deep_in = emb_dim * len(cat_cols) + num_dim
        layers = []
        prev = deep_in
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.deep_mlp = nn.Sequential(*layers)

    def forward(self, wide_idx: torch.Tensor, deep_cat_idx: torch.Tensor, num_x: torch.Tensor) -> torch.Tensor:
        # wide_idx: [B, K_cat]
        if wide_idx.numel() > 0:
            wide_w = self.wide_emb(wide_idx)  # [B, K_cat, 1]
            wide_logit = wide_w.sum(dim=1)    # [B, 1]
        else:
            wide_logit = torch.zeros((num_x.size(0), 1), device=num_x.device)

        # deep part
        deep_parts = []
        if deep_cat_idx.numel() > 0:
            # deep_cat_idx: [B, K_cat]
            for j, c in enumerate(self.cat_cols):
                emb = self.cat_embeddings[c](deep_cat_idx[:, j])  # [B, E]
                deep_parts.append(emb)
        if num_x.numel() > 0:
            deep_parts.append(num_x)
        deep_in = torch.cat(deep_parts, dim=1) if len(deep_parts) > 0 else torch.zeros((num_x.size(0), 0), device=num_x.device)
        deep_logit = self.deep_mlp(deep_in)  # [B,1]

        return wide_logit + deep_logit


# -------------------------
# Train & Eval
# -------------------------

def build_val_frames(val_df: pd.DataFrame, scores: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    tmp = val_df.copy()
    tmp["final_score"] = scores

    g = tmp.groupby("session_id")
    ordered_items = g.apply(lambda grp: " ".join(grp.loc[grp["ordered"] == 1, "content_id_hashed"].astype(str)))
    clicked_items = g.apply(lambda grp: " ".join(grp.loc[grp["clicked"] == 1, "content_id_hashed"].astype(str)))
    all_items = g["content_id_hashed"].apply(lambda s: " ".join(s.astype(str)))

    solution = (
        pd.DataFrame({
            "ordered_items": ordered_items,
            "clicked_items": clicked_items,
            "all_items": all_items,
        })
        .reset_index()
    )

    submission = (
        tmp.sort_values(["session_id", "final_score"], ascending=[True, False])
        .groupby("session_id")["content_id_hashed"]
        .apply(lambda s: " ".join(s.astype(str)))
        .reset_index()
        .rename(columns={"content_id_hashed": "prediction"})
    )
    return solution, submission


def train_loop(model: WideAndDeep, train_loader: DataLoader, val_loader: DataLoader, device: torch.device,
               epochs: int = 3, lr: float = 1e-3, accumulate_steps: int = 1) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss(reduction='none')
    scaler = GradScaler(enabled=torch.cuda.is_available())

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n = 0
        opt.zero_grad(set_to_none=True)
        step = 0
        for wide_idx, deep_cat_idx, num_x, y, w in train_loader:
            wide_idx = wide_idx.to(device, non_blocking=True)
            deep_cat_idx = deep_cat_idx.to(device, non_blocking=True)
            num_x = num_x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).view(-1, 1)
            w = w.to(device, non_blocking=True).view(-1, 1)

            with autocast(enabled=torch.cuda.is_available()):
                logits = model(wide_idx, deep_cat_idx, num_x)
                loss = bce(logits, y)
                loss = (loss * w).mean()

            scaler.scale(loss).backward()
            step += 1
            if step % max(1, accumulate_steps) == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            total_loss += float(loss.detach().item()) * y.size(0)
            n += y.size(0)
        _print(f"epoch {epoch}: train_loss={total_loss / max(1,n):.6f}")

        # quick val AUC (overall)
        model.eval()
        ys = []
        ps = []
        with torch.no_grad():
            for wide_idx, deep_cat_idx, num_x, y, _ in val_loader:
                with autocast(enabled=torch.cuda.is_available()):
                    logits = model(wide_idx.to(device, non_blocking=True), deep_cat_idx.to(device, non_blocking=True), num_x.to(device, non_blocking=True))
                    p = torch.sigmoid(logits).squeeze(1).float().cpu().numpy()
                ps.append(p)
                ys.append(y.numpy())
        y_true = np.concatenate(ys) if ys else np.array([])
        p_all = np.concatenate(ps) if ps else np.array([])
        try:
            auc = roc_auc_score(y_true, p_all) if y_true.size and len(set(y_true))>1 else float('nan')
        except Exception:
            auc = float('nan')
        _print(f"epoch {epoch}: val_auc={auc:.6f}")


# collate function to pad absent lists (but here lengths are fixed by #cat_cols)

def collate(batch):
    wide_idx = [b[0] for b in batch]
    deep_idx = [b[1] for b in batch]
    num_x = [b[2] for b in batch]
    y = [b[3] for b in batch]
    w = [b[4] for b in batch]
    wide_idx = torch.as_tensor(np.stack(wide_idx, axis=0), dtype=torch.long) if wide_idx[0].size > 0 else torch.zeros((len(batch),0), dtype=torch.long)
    deep_idx = torch.as_tensor(np.stack(deep_idx, axis=0), dtype=torch.long) if deep_idx[0].size > 0 else torch.zeros((len(batch),0), dtype=torch.long)
    num_x = torch.as_tensor(np.stack(num_x, axis=0), dtype=torch.float32) if num_x[0].size > 0 else torch.zeros((len(batch),0), dtype=torch.float32)
    y = torch.as_tensor(np.array(y), dtype=torch.float32)
    w = torch.as_tensor(np.array(w), dtype=torch.float32)
    return wide_idx, deep_idx, num_x, y, w


# -------------------------
# Main
# -------------------------

def main(root: Path, w_order: float, w_click: float, val_frac: float,
         epochs: int, batch_size: int, lr: float, emb_dim: int, wide_hash_dim: int):
    models_dir = root / "models" / "widedeep"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Load
    train_cands = [root / "veri" / "train_sessions.parquet", root / "veri" / "enriched_train_data_v2.parquet"]
    train_path = next((p for p in train_cands if p.exists()), None)
    if train_path is None:
        raise FileNotFoundError("Train parquet bulunamadı (veri/train_sessions.parquet veya enriched_*).")

    _print(f"Loading: {train_path}")
    df_pl = pl.read_parquet(str(train_path))
    train_pd, val_pd = time_based_split(df_pl, val_frac=val_frac)

    # Features
    cat_cols, num_cols = build_features(train_pd)
    _print(f"n_cat={len(cat_cols)}, n_num={len(num_cols)}")

    # Cat buckets per feature
    cat_bucket_sizes = {c: 50000 for c in cat_cols}

    # Numeric stats
    num_means = {c: float(train_pd[c].mean()) for c in num_cols}
    num_stds = {c: float(train_pd[c].std(ddof=0) or 1.0) for c in num_cols}

    # Datasets
    ds_tr = WideDeepDataset(train_pd, cat_cols, num_cols, w_order, w_click, num_means, num_stds, cat_bucket_sizes, wide_hash_dim)
    ds_va = WideDeepDataset(val_pd, cat_cols, num_cols, w_order, w_click, num_means, num_stds, cat_bucket_sizes, wide_hash_dim)
    # DataLoader tuning for better GPU utilization
    cpu_cnt = os.cpu_count() or 1
    workers = max(2, min(8, max(1, cpu_cnt // 2)))
    dl_tr = DataLoader(
        ds_tr,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        collate_fn=collate,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    dl_va = DataLoader(
        ds_va,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=collate,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = WideAndDeep(cat_cols, num_dim=len(num_cols), cat_bucket_sizes=cat_bucket_sizes,
                        wide_hash_dim=wide_hash_dim, emb_dim=emb_dim).to(device)

    # Enable CuDNN autotune when input sizes are stable
    torch.backends.cudnn.benchmark = True
    train_loop(model, dl_tr, dl_va, device, epochs=epochs, lr=lr, accumulate_steps=max(1, 262144 // max(1, batch_size)))

    # Validation (competition metric)
    model.eval()
    with torch.no_grad():
        scores = []
        for wide_idx, deep_cat_idx, num_x, _, _ in dl_va:
            with autocast(enabled=torch.cuda.is_available()):
                logits = model(wide_idx.to(device, non_blocking=True), deep_cat_idx.to(device, non_blocking=True), num_x.to(device, non_blocking=True))
                scores.append(torch.sigmoid(logits).squeeze(1).float().cpu().numpy())
    val_scores = np.concatenate(scores) if scores else np.array([])
    sol, sub = build_val_frames(val_pd, val_scores)
    final_score = group_auc_score(sol, sub, "session_id")
    _print(f"Validation Final (metric): {final_score:.6f}")

    # Save model & meta
    model_path = models_dir / "widedeep.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "cat_bucket_sizes": cat_bucket_sizes,
        "wide_hash_dim": wide_hash_dim,
        "emb_dim": emb_dim,
        "num_means": num_means,
        "num_stds": num_stds,
        "weights": {"ordered": w_order, "clicked": w_click},
    }, model_path)

    meta = {
        "model_path": model_path.as_posix(),
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "cat_bucket_sizes": cat_bucket_sizes,
        "wide_hash_dim": wide_hash_dim,
        "emb_dim": emb_dim,
        "num_means": num_means,
        "num_stds": num_stds,
        "validation_final": final_score,
    }
    (models_dir / "meta_widedeep.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _print(f"Saved Wide&Deep model to {model_path}")

    del model
    gc.collect()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default=None, help="Proje kök dizini; varsayılan bu dosyanın klasörü")
    p.add_argument("--w_order", type=float, default=0.7)
    p.add_argument("--w_click", type=float, default=0.3)
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=8192)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--emb_dim", type=int, default=16)
    p.add_argument("--wide_hash_dim", type=int, default=200000)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root = Path(args.root) if args.root else Path(__file__).parent
    main(
        root=root,
        w_order=args.w_order,
        w_click=args.w_click,
        val_frac=args.val_frac,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        emb_dim=args.emb_dim,
        wide_hash_dim=args.wide_hash_dim,
    )
