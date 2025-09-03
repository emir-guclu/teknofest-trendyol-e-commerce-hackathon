import argparse
import json
import os
from pathlib import Path
from typing import List

# Pin to a single GPU before importing CatBoost
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import gc
import numpy as np
import pandas as pd
import polars as pl
from catboost import CatBoostClassifier


def build_features(df: pd.DataFrame) -> List[str]:
    targets = ["ordered", "clicked"]
    exclude = set(targets + [
        "ts_hour",
        "session_id",
        "content_creation_date",
        "update_date",
        "added_to_cart",
        "added_to_fav",
    ])
    return [c for c in df.columns if c not in exclude]


def infer_cat_features(df: pd.DataFrame, features: List[str]) -> List[str]:
    cats: List[str] = []
    for c in features:
        dt = str(df[c].dtype)
        if dt in ("object", "string", "string[python]"):
            cats.append(c)
    return cats


def main(root: Path):
    models_dir = root / "models" / "catboost_cls"
    meta_path = models_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Bulunamadı: {meta_path}. Önce HPO çalıştırıp en iyi parametreleri kaydedin.")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    best_params = meta.get("best_params", {})
    meta_features = meta.get("features", [])
    meta_cats = meta.get("cat_features", [])

    # Load training data (use enriched if available)
    train_cands = [root / "veri" / "train_sessions.parquet", root / "veri" / "enriched_train_data_v2.parquet"]
    train_path = next((p for p in train_cands if p.exists()), None)
    if train_path is None:
        raise FileNotFoundError("Train parquet dosyası bulunamadı (veri/train_sessions.parquet veya enriched_*).")

    df_pl = pl.read_parquet(str(train_path))
    df = df_pl.to_pandas()

    # Determine features from meta; fallback to rebuild and intersect
    features = [f for f in (meta_features or build_features(df)) if f in df.columns]
    cat_features = [c for c in (meta_cats or infer_cat_features(df, features)) if c in features]

    # Targets
    if not {"ordered", "clicked"}.issubset(df.columns):
        raise RuntimeError("Eğitim için 'ordered' ve 'clicked' kolonları gerekli.")
    y_order = df["ordered"]
    y_click = df["clicked"]

    # Class imbalance weights
    order_spw = max(1, int((y_order == 0).sum())) / max(1, int(y_order.sum()))
    click_spw = max(1, int((y_click == 0).sum())) / max(1, int(y_click.sum()))

    # Build model params from meta
    iterations = int(best_params.get("iterations", 800))
    depth = int(best_params.get("depth", 6))
    learning_rate = float(best_params.get("learning_rate", 0.1))
    l2_leaf_reg = float(best_params.get("l2_leaf_reg", 3.0))
    subsample = float(best_params.get("subsample", 0.8))

    base_params = dict(
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        l2_leaf_reg=l2_leaf_reg,
        subsample=subsample,
        bootstrap_type="Bernoulli",
        eval_metric="AUC",
        task_type="GPU",
        devices="0",
        verbose=100,
        random_seed=42,
    )

    X = df[features]

    # Train ordered head on all data
    m_o = CatBoostClassifier(**{**base_params, "scale_pos_weight": order_spw, "cat_features": cat_features})
    m_o.fit(X, y_order)
    ordered_out = models_dir / "model_ordered_full.cbm"
    m_o.save_model(str(ordered_out))
    del m_o; gc.collect()

    # Train clicked head on all data
    m_c = CatBoostClassifier(**{**base_params, "scale_pos_weight": click_spw, "cat_features": cat_features})
    m_c.fit(X, y_click)
    clicked_out = models_dir / "model_clicked_full.cbm"
    m_c.save_model(str(clicked_out))
    del m_c; gc.collect()

    meta_full = {
        "source_meta": meta,
        "trained_on": "train_sessions_full",
        "features": features,
        "cat_features": cat_features,
        "model_paths": {
            "ordered": ordered_out.as_posix(),
            "clicked": clicked_out.as_posix(),
        },
    }
    (models_dir / "meta_full.json").write_text(json.dumps(meta_full, indent=2), encoding="utf-8")
    print("Tam eğitim tamamlandı. Kaydedildi:", ordered_out, clicked_out)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default=None, help="Proje kök dizini (varsayılan: bu dosyanın klasörü)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root = Path(args.root) if args.root else Path(__file__).parent
    main(root)
