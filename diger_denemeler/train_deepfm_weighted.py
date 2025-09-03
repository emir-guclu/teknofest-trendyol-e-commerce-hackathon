import argparse
import json
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
    print(f"[deepfm] {msg}")


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


def stable_hash(s: str) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)


# -------------------------
# Dataset
# -------------------------

class DeepFMDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cat_cols: List[str], num_cols: List[str],
                 w_order: float, w_click: float,
                 num_means: Dict[str, float] | None = None,
                 num_stds: Dict[str, float] | None = None,
                 cat_bucket_sizes: Dict[str, int] | None = None):
        self.df = df.reset_index(drop=True)
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.w_order = w_order
        self.w_click = w_click
        self.num_means = num_means or {c: float(self.df[c].mean()) for c in num_cols}
        self.num_stds = num_stds or {c: float(self.df[c].std(ddof=0) or 1.0) for c in num_cols}
        self.cat_bucket_sizes = cat_bucket_sizes or {c: 50000 for c in cat_cols}

        # Numeric matrix
        self.num_arr = None
        if num_cols:
            arr = []
            for c in num_cols:
                m, s = self.num_means[c], self.num_stds[c] or 1.0
                x = self.df[c].astype(float).fillna(m).to_numpy()
                arr.append((x - m) / (s if s != 0 else 1.0))
            self.num_arr = np.stack(arr, axis=1).astype(np.float32)

        # Labels and weights
        self.y = ((self.df["ordered"] == 1) | (self.df["clicked"] == 1)).astype(np.float32).to_numpy()
        rel = (self.w_order * self.df["ordered"].astype(float) + self.w_click * self.df["clicked"].astype(float)).to_numpy()
        pos_count = float(self.y.sum())
        neg_count = float((self.y == 0).sum())
        # If there are no negatives (e.g., positive-only training), avoid collapsing weights
        base = (neg_count / max(1.0, pos_count)) if neg_count > 0 else 1.0
        self.w = np.where(self.y == 1.0, base * rel, 1.0).astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        # hashed categorical indices per feature
        cat_idx = []
        for c in self.cat_cols:
            v = str(row[c])
            bucket = self.cat_bucket_sizes[c]
            h = stable_hash(f"{c}={v}") % bucket
            cat_idx.append(h)
        cat_idx = np.array(cat_idx, dtype=np.int64) if self.cat_cols else np.zeros((0,), dtype=np.int64)
        num_x = self.num_arr[idx] if self.num_arr is not None else np.zeros((0,), dtype=np.float32)
        return cat_idx, num_x, self.y[idx], self.w[idx]


def collate(batch):
    cat_idx = [b[0] for b in batch]
    num_x = [b[1] for b in batch]
    y = [b[2] for b in batch]
    w = [b[3] for b in batch]
    cat_idx = torch.as_tensor(np.stack(cat_idx, axis=0), dtype=torch.long) if cat_idx[0].size > 0 else torch.zeros((len(batch),0), dtype=torch.long)
    num_x = torch.as_tensor(np.stack(num_x, axis=0), dtype=torch.float32) if num_x[0].size > 0 else torch.zeros((len(batch),0), dtype=torch.float32)
    y = torch.as_tensor(np.array(y), dtype=torch.float32)
    w = torch.as_tensor(np.array(w), dtype=torch.float32)
    return cat_idx, num_x, y, w


# -------------------------
# DeepFM Model
# -------------------------

class DeepFM(nn.Module):
    def __init__(self, cat_cols: List[str], num_dim: int, cat_bucket_sizes: Dict[str, int], emb_dim: int = 16, hidden_dims: List[int] = [256,128,64], dropout: float = 0.1):
        super().__init__()
        self.cat_cols = cat_cols
        self.num_dim = num_dim
        self.emb_dim = emb_dim

        # Linear part
        self.linear_cat = nn.ModuleDict({c: nn.Embedding(cat_bucket_sizes[c], 1) for c in cat_cols})
        self.linear_num = nn.Linear(num_dim, 1) if num_dim > 0 else None

        # FM part (second-order) embeddings
        self.fm_embeddings = nn.ModuleDict({c: nn.Embedding(cat_bucket_sizes[c], emb_dim) for c in cat_cols})

        # Deep part
        deep_in = emb_dim * len(cat_cols) + num_dim
        layers = []
        prev = deep_in
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.deep_mlp = nn.Sequential(*layers)

    def fm_second_order(self, emb_list: List[torch.Tensor]) -> torch.Tensor:
        # emb_list: list of [B, K]
        if not emb_list:
            return torch.zeros((emb_list[0].size(0) if emb_list else 1, 1), device=next(self.parameters()).device)
        E = torch.stack(emb_list, dim=1)  # [B, F, K]
        sum_square = E.sum(dim=1) ** 2      # [B, K]
        square_sum = (E ** 2).sum(dim=1)    # [B, K]
        second = 0.5 * (sum_square - square_sum).sum(dim=1, keepdim=True)  # [B,1]
        return second

    def forward(self, cat_idx: torch.Tensor, num_x: torch.Tensor) -> torch.Tensor:
        B = cat_idx.size(0)
        device = cat_idx.device if cat_idx.numel() > 0 else num_x.device
        # Linear part
        linear_terms = []
        if self.cat_cols:
            for j, c in enumerate(self.cat_cols):
                linear_terms.append(self.linear_cat[c](cat_idx[:, j]))  # [B,1]
        if self.linear_num is not None:
            linear_terms.append(self.linear_num(num_x))
        linear_logit = torch.stack(linear_terms, dim=0).sum(dim=0) if linear_terms else torch.zeros((B,1), device=device)

        # FM part (second-order on cat embeddings)
        fm_embs = []
        if self.cat_cols:
            for j, c in enumerate(self.cat_cols):
                fm_embs.append(self.fm_embeddings[c](cat_idx[:, j]))  # [B,K]
        fm_logit = self.fm_second_order(fm_embs) if fm_embs else torch.zeros((B,1), device=device)

        # Deep part
        deep_parts = []
        if fm_embs:
            deep_parts.append(torch.cat(fm_embs, dim=1))  # [B, F*K]
        if num_x.numel() > 0:
            deep_parts.append(num_x)
        deep_in = torch.cat(deep_parts, dim=1) if deep_parts else torch.zeros((B,0), device=device)
        deep_logit = self.deep_mlp(deep_in)  # [B,1]

        return linear_logit + fm_logit + deep_logit


# -------------------------
# Train & Eval
# -------------------------

def build_val_frames(val_df: pd.DataFrame, scores: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    tmp = val_df.copy(); tmp["final_score"] = scores
    g = tmp.groupby("session_id")
    solution = pd.DataFrame({
        "ordered_items": g.apply(lambda grp: " ".join(grp.loc[grp["ordered"]==1, "content_id_hashed"].astype(str))),
        "clicked_items": g.apply(lambda grp: " ".join(grp.loc[grp["clicked"]==1, "content_id_hashed"].astype(str))),
        "all_items": g["content_id_hashed"].apply(lambda s: " ".join(s.astype(str))),
    }).reset_index()
    submission = (
        tmp.sort_values(["session_id","final_score"], ascending=[True,False])
        .groupby("session_id")["content_id_hashed"].apply(lambda s: " ".join(s.astype(str)))
        .reset_index().rename(columns={"content_id_hashed":"prediction"})
    )
    return solution, submission


def train_loop(model: DeepFM, train_loader: DataLoader, val_loader: DataLoader, device: torch.device, epochs: int = 3, lr: float = 1e-3, accumulate_steps: int = 1):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss(reduction='none')
    scaler = GradScaler(enabled=torch.cuda.is_available())

    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0; n = 0; step = 0
        opt.zero_grad(set_to_none=True)
        for cat_idx, num_x, y, w in train_loader:
            cat_idx = cat_idx.to(device, non_blocking=True)
            num_x = num_x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).view(-1,1)
            w = w.to(device, non_blocking=True).view(-1,1)
            with autocast(enabled=torch.cuda.is_available()):
                logits = model(cat_idx, num_x)
                loss = bce(logits, y)
                loss = (loss * w).mean()
            scaler.scale(loss).backward()
            step += 1
            if step % max(1,accumulate_steps) == 0:
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            total_loss += float(loss.detach().item()) * y.size(0)
            n += y.size(0)
        _print(f"epoch {epoch}: train_loss={total_loss/max(1,n):.6f}")

        model.eval(); ys=[]; ps=[]
        with torch.no_grad():
            for cat_idx, num_x, y, _ in val_loader:
                with autocast(enabled=torch.cuda.is_available()):
                    logits = model(cat_idx.to(device, non_blocking=True), num_x.to(device, non_blocking=True))
                    p = torch.sigmoid(logits).squeeze(1).float().cpu().numpy()
                ps.append(p); ys.append(y.numpy())
        y_true = np.concatenate(ys) if ys else np.array([])
        p_all = np.concatenate(ps) if ps else np.array([])
        try:
            auc = roc_auc_score(y_true, p_all) if y_true.size and len(set(y_true))>1 else float('nan')
        except Exception:
            auc = float('nan')
        _print(f"epoch {epoch}: val_auc={auc:.6f}")


# -------------------------
# Main
# -------------------------

def main(root: Path, w_order: float, w_click: float, val_frac: float, epochs: int, batch_size: int, lr: float, emb_dim: int):
    models_dir = root / "models" / "deepfm"; models_dir.mkdir(parents=True, exist_ok=True)

    # Load
    train_cands = [root / "veri" / "train_sessions.parquet", root / "veri" / "enriched_train_data_v2.parquet"]
    train_path = next((p for p in train_cands if p.exists()), None)
    if train_path is None:
        raise FileNotFoundError("Train parquet bulunamadı (veri/train_sessions.parquet veya enriched_*).")

    _print(f"Loading: {train_path}")
    df_pl = pl.read_parquet(str(train_path))
    train_pd, val_pd = time_based_split(df_pl, val_frac=val_frac)
    # Filter out both_zero from training set
    both_zero_mask = (train_pd["ordered"] == 0) & (train_pd["clicked"] == 0)
    train_pd_pos = train_pd.loc[~both_zero_mask].reset_index(drop=True)
    _print(f"train rows: {len(train_pd)}, after filtering both_zero: {len(train_pd_pos)}")

    # Features & stats
    cat_cols, num_cols = build_features(train_pd)
    _print(f"n_cat={len(cat_cols)}, n_num={len(num_cols)}")
    cat_bucket_sizes = {c: 50000 for c in cat_cols}
    num_means = {c: float(train_pd[c].mean()) for c in num_cols}
    num_stds = {c: float(train_pd[c].std(ddof=0) or 1.0) for c in num_cols}

    # Datasets & Loaders
    ds_tr = DeepFMDataset(train_pd_pos, cat_cols, num_cols, w_order, w_click, num_means, num_stds, cat_bucket_sizes)
    ds_va = DeepFMDataset(val_pd, cat_cols, num_cols, w_order, w_click, num_means, num_stds, cat_bucket_sizes)
    cpu_cnt = os.cpu_count() or 1
    # Windows DataLoader workers can error in some setups; fallback to single-process there
    workers = 0 if os.name == 'nt' else max(2, min(8, max(1, cpu_cnt // 2)))
    if workers > 0:
        dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=workers, collate_fn=collate, pin_memory=True, persistent_workers=True, prefetch_factor=2)
        dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=workers, collate_fn=collate, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    else:
        dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate, pin_memory=True)
        dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    model = DeepFM(cat_cols, num_dim=len(num_cols), cat_bucket_sizes=cat_bucket_sizes, emb_dim=emb_dim).to(device)
    train_loop(model, dl_tr, dl_va, device, epochs=epochs, lr=lr, accumulate_steps=max(1, 262144 // max(1, batch_size)))

    # Validation metric (competition)
    model.eval(); scores=[]
    with torch.no_grad():
        for cat_idx, num_x, _, _ in dl_va:
            with autocast(enabled=torch.cuda.is_available()):
                logits = model(cat_idx.to(device, non_blocking=True), num_x.to(device, non_blocking=True))
                scores.append(torch.sigmoid(logits).squeeze(1).float().cpu().numpy())
    val_scores = np.concatenate(scores) if scores else np.array([])
    sol, sub = build_val_frames(val_pd, val_scores)
    final_score = group_auc_score(sol, sub, "session_id")
    _print(f"Validation Final (metric): {final_score:.6f}")

    # Save
    model_path = models_dir / "deepfm.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "cat_bucket_sizes": cat_bucket_sizes,
        "num_means": num_means,
        "num_stds": num_stds,
        "emb_dim": emb_dim,
        "weights": {"ordered": w_order, "clicked": w_click},
    }, model_path)
    meta = {
        "model_path": model_path.as_posix(),
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "cat_bucket_sizes": cat_bucket_sizes,
        "num_means": num_means,
        "num_stds": num_stds,
        "emb_dim": emb_dim,
        "validation_final": final_score,
    }
    (models_dir / "meta_deepfm.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _print(f"Saved DeepFM model to {model_path}")

    del model
    gc.collect()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default=None)
    p.add_argument("--w_order", type=float, default=0.7)
    p.add_argument("--w_click", type=float, default=0.3)
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=65536)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--emb_dim", type=int, default=16)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root = Path(args.root) if args.root else Path(__file__).parent
    main(root, w_order=args.w_order, w_click=args.w_click, val_frac=args.val_frac, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, emb_dim=args.emb_dim)
