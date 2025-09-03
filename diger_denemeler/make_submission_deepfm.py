import argparse
import json
from pathlib import Path
import os
import hashlib
import numpy as np
import pandas as pd
import polars as pl
import torch
from torch.cuda.amp import autocast


def _print(msg: str):
    print(f"[deepfm-sub] {msg}")


def stable_hash(s: str) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)


def build_features(df: pd.DataFrame, meta: dict):
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
        if str(df[c].dtype).startswith("datetime") or any(k in c.lower() for k in time_name_keywords):
            exclude.add(c)
    cols = [c for c in df.columns if c not in exclude]
    cat_cols = [c for c in cols if c in meta["cat_cols"]]
    num_cols = [c for c in cols if c in meta["num_cols"]]
    return cat_cols, num_cols


def main(root: Path):
    model_path = root / "models" / "deepfm" / "deepfm.pt"
    meta_path = root / "models" / "deepfm" / "meta_deepfm.json"
    if not model_path.exists():
        raise FileNotFoundError("deepfm.pt bulunamadı, önce eğitin.")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    train_cands = [root / "veri" / "train_sessions.parquet", root / "veri" / "enriched_train_data_v2.parquet"]
    test_cands = [root / "veri" / "test_sessions.parquet", root / "veri" / "enriched_test_data_v2.parquet"]
    test_path = next((p for p in test_cands if p.exists()), None)
    if test_path is None:
        raise FileNotFoundError("Test parquet bulunamadı (veri/test_sessions.parquet veya enriched_*).")

    _print(f"Loading test: {test_path}")
    df_pl = pl.read_parquet(str(test_path))
    test_pd = df_pl.to_pandas()

    # Load torch model
    ckpt = torch.load(model_path, map_location="cpu")
    cat_cols = ckpt["cat_cols"]; num_cols = ckpt["num_cols"]
    cat_bucket_sizes = ckpt["cat_bucket_sizes"]; num_means = ckpt["num_means"]; num_stds = ckpt["num_stds"]

    class DeepFM(torch.nn.Module):
        def __init__(self, cat_cols, num_dim, cat_bucket_sizes, emb_dim):
            super().__init__()
            self.cat_cols = cat_cols
            self.linear_cat = torch.nn.ModuleDict({c: torch.nn.Embedding(cat_bucket_sizes[c], 1) for c in cat_cols})
            self.linear_num = torch.nn.Linear(num_dim, 1) if num_dim>0 else None
            self.fm_embeddings = torch.nn.ModuleDict({c: torch.nn.Embedding(cat_bucket_sizes[c], emb_dim) for c in cat_cols})
            deep_in = emb_dim*len(cat_cols) + num_dim
            layers = []
            prev = deep_in
            for h in [256,128,64]:
                layers += [torch.nn.Linear(prev, h), torch.nn.ReLU(), torch.nn.Dropout(0.1)]
                prev = h
            layers.append(torch.nn.Linear(prev, 1))
            self.deep_mlp = torch.nn.Sequential(*layers)
        def fm_second_order(self, emb_list):
            if not emb_list:
                return torch.zeros((emb_list[0].size(0) if emb_list else 1,1), device=next(self.parameters()).device)
            E = torch.stack(emb_list, dim=1)
            sum_square = E.sum(dim=1) ** 2
            square_sum = (E ** 2).sum(dim=1)
            return 0.5 * (sum_square - square_sum).sum(dim=1, keepdim=True)
        def forward(self, cat_idx, num_x):
            B = cat_idx.size(0)
            device = cat_idx.device if cat_idx.numel()>0 else num_x.device
            linear_terms = []
            if self.cat_cols:
                for j,c in enumerate(self.cat_cols):
                    linear_terms.append(self.linear_cat[c](cat_idx[:, j]))
            if self.linear_num is not None:
                linear_terms.append(self.linear_num(num_x))
            linear_logit = torch.stack(linear_terms, dim=0).sum(dim=0) if linear_terms else torch.zeros((B,1), device=device)
            fm_embs = []
            if self.cat_cols:
                for j,c in enumerate(self.cat_cols):
                    fm_embs.append(self.fm_embeddings[c](cat_idx[:, j]))
            fm_logit = self.fm_second_order(fm_embs) if fm_embs else torch.zeros((B,1), device=device)
            deep_parts = []
            if fm_embs:
                deep_parts.append(torch.cat(fm_embs, dim=1))
            if num_x.numel()>0:
                deep_parts.append(num_x)
            deep_in = torch.cat(deep_parts, dim=1) if deep_parts else torch.zeros((B,0), device=device)
            deep_logit = self.deep_mlp(deep_in)
            return linear_logit + fm_logit + deep_logit

    model = DeepFM(cat_cols, num_dim=len(num_cols), cat_bucket_sizes=cat_bucket_sizes, emb_dim=ckpt.get("emb_dim",16))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Build features per meta
    def transform(df: pd.DataFrame):
        cat_idx = []
        for c in cat_cols:
            v = df[c].astype(str).fillna("<na>")
            h = v.map(lambda s: stable_hash(f"{c}={s}") % cat_bucket_sizes[c])
            cat_idx.append(h.values)
        cat_idx = np.stack(cat_idx, axis=1).astype(np.int64) if cat_cols else np.zeros((len(df),0), dtype=np.int64)
        num_x = []
        for c in num_cols:
            m = num_means[c]; s = num_stds[c] or 1.0
            x = df[c].astype(float).fillna(m).to_numpy()
            num_x.append((x - m) / (s if s!=0 else 1.0))
        num_x = np.stack(num_x, axis=1).astype(np.float32) if num_cols else np.zeros((len(df),0), dtype=np.float32)
        return cat_idx, num_x

    cat_idx, num_x = transform(test_pd)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    scores = []
    B = 131072
    for i in range(0, len(test_pd), B):
        ci = torch.as_tensor(cat_idx[i:i+B], dtype=torch.long, device=device, non_blocking=True)
        nx = torch.as_tensor(num_x[i:i+B], dtype=torch.float32, device=device, non_blocking=True)
        with torch.no_grad():
            with autocast(enabled=torch.cuda.is_available()):
                logits = model(ci, nx)
                p = torch.sigmoid(logits).squeeze(1).float().cpu().numpy()
        scores.append(p)
    scores = np.concatenate(scores) if scores else np.array([])

    tmp = test_pd.copy(); tmp["final_score"] = scores
    submission = (
        tmp.sort_values(["session_id","final_score"], ascending=[True,False])
        .groupby("session_id")["content_id_hashed"].apply(lambda s: " ".join(s.astype(str)))
        .reset_index().rename(columns={"content_id_hashed":"prediction"})
    )

    out_path = root / "submission_deepfm.csv"
    submission.to_csv(out_path, index=False)
    _print(f"Saved submission to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=None)
    args = ap.parse_args()
    root = Path(args.root) if args.root else Path(__file__).parent
    main(root)
