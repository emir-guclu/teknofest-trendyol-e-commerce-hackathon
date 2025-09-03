import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import gc
import numpy as np
import pandas as pd
import polars as pl
import torch
from torch import nn


def _print(msg: str):
    print(f"[submit_widedeep] {msg}")


def build_features(df: pd.DataFrame):
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


class WideAndDeep(nn.Module):
    def __init__(self, cat_cols, num_dim, cat_bucket_sizes, wide_hash_dim, emb_dim):
        super().__init__()
        self.cat_cols = cat_cols
        self.num_dim = num_dim
        self.wide_hash_dim = wide_hash_dim
        self.wide_emb = nn.Embedding(wide_hash_dim, 1)
        self.cat_embeddings = nn.ModuleDict({c: nn.Embedding(cat_bucket_sizes[c], emb_dim) for c in cat_cols})
        deep_in = emb_dim * len(cat_cols) + num_dim
        self.deep_mlp = nn.Sequential(
            nn.Linear(deep_in, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 1),
        )
    def forward(self, wide_idx, deep_cat_idx, num_x):
        if wide_idx.numel() > 0:
            wide_w = self.wide_emb(wide_idx)
            wide_logit = wide_w.sum(dim=1)
        else:
            wide_logit = torch.zeros((num_x.size(0), 1), device=num_x.device)
        deep_parts = []
        if deep_cat_idx.numel() > 0:
            for j, c in enumerate(self.cat_cols):
                deep_parts.append(self.cat_embeddings[c](deep_cat_idx[:, j]))
        if num_x.numel() > 0:
            deep_parts.append(num_x)
        deep_in = torch.cat(deep_parts, dim=1) if deep_parts else torch.zeros((num_x.size(0), 0), device=num_x.device)
        deep_logit = self.deep_mlp(deep_in)
        return wide_logit + deep_logit


def stable_hash(s: str) -> int:
    import hashlib
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)


def main(root: Path, out_name: str = "submission_widedeep.csv"):
    models_dir = root / "models" / "widedeep"
    meta_p = models_dir / "meta_widedeep.json"
    if not meta_p.exists():
        raise FileNotFoundError(f"{meta_p} bulunamadı. Önce train_widedeep_weighted.py ile model eğitin.")

    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    model_p = Path(meta["model_path"]) if isinstance(meta.get("model_path"), str) else (models_dir / "widedeep.pt")

    # Load test
    test_cands = [root / "veri" / "test_sessions.parquet", root / "veri" / "enriched_test_data_v2.parquet"]
    test_path = next((p for p in test_cands if p.exists()), None)
    if test_path is None:
        raise FileNotFoundError("Test parquet dosyası bulunamadı (veri/test_sessions.parquet veya enriched_*).")

    test_pl = pl.read_parquet(str(test_path))
    test_df = test_pl.to_pandas()

    cat_cols, num_cols = build_features(test_df)

    # Prepare hashed inputs
    wide_hash_dim = int(meta["wide_hash_dim"])
    emb_dim = int(meta["emb_dim"])
    cat_bucket_sizes = {k: int(v) for k, v in meta["cat_bucket_sizes"].items()}

    def to_batch(df: pd.DataFrame):
        wide_idx = []
        deep_idx = []
        num_x = []
        for _, row in df.iterrows():
            wi = []
            di = []
            for c in cat_cols:
                v = str(row[c])
                wi.append(stable_hash(f"{c}={v}") % wide_hash_dim)
                di.append(stable_hash(f"{c}={v}") % cat_bucket_sizes[c])
            wide_idx.append(wi)
            deep_idx.append(di)
            if num_cols:
                num_x.append(row[num_cols].astype(float).to_numpy())
            else:
                num_x.append(np.zeros((0,), dtype=np.float32))
        return (
            torch.as_tensor(np.array(wide_idx), dtype=torch.long) if cat_cols else torch.zeros((len(df),0), dtype=torch.long),
            torch.as_tensor(np.array(deep_idx), dtype=torch.long) if cat_cols else torch.zeros((len(df),0), dtype=torch.long),
            torch.as_tensor(np.array(num_x), dtype=torch.float32) if num_cols else torch.zeros((len(df),0), dtype=torch.float32),
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(model_p, map_location=device)
    model = WideAndDeep(
        cat_cols=ckpt["cat_cols"],
        num_dim=len(ckpt["num_cols"]),
        cat_bucket_sizes=ckpt["cat_bucket_sizes"],
        wide_hash_dim=int(ckpt["wide_hash_dim"]),
        emb_dim=int(ckpt["emb_dim"]),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    wide_idx, deep_idx, num_x = to_batch(test_df)
    with torch.no_grad():
        logits = model(wide_idx.to(device), deep_idx.to(device), num_x.to(device))
        scores = torch.sigmoid(logits).squeeze(1).cpu().numpy()

    test_df['final_score'] = scores
    sub = (
        test_df.sort_values(['session_id', 'final_score'], ascending=[True, False])
        .groupby('session_id')['content_id_hashed']
        .apply(lambda s: ' '.join(s.astype(str)))
        .reset_index()
        .rename(columns={'content_id_hashed': 'prediction'})
    )

    out_path = root / out_name
    sub.to_csv(out_path, index=False)
    _print(f"Submission kaydedildi: {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default=None, help="Proje kök dizini (varsayılan: bu dosyanın klasörü)")
    p.add_argument("--out", type=str, default="submission_widedeep.csv", help="Çıktı dosya adı")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root = Path(args.root) if args.root else Path(__file__).parent
    main(root, out_name=args.out)
