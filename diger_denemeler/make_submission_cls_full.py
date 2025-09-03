import argparse
import json
import os
from pathlib import Path

# Single GPU
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import gc
import pandas as pd
import polars as pl
from catboost import CatBoostClassifier


def build_features(df: pd.DataFrame):
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


def main(root: Path, out_name: str = "submission_cls_full.csv"):
    models_dir = root / "models" / "catboost_cls"
    meta_p = models_dir / "meta.json"
    meta_full_p = models_dir / "meta_full.json"

    if not meta_full_p.exists():
        raise FileNotFoundError(f"{meta_full_p} bulunamadı. Önce train_catboost_cls_full.py ile full eğitim yapın.")
    if not meta_p.exists():
        raise FileNotFoundError(f"{meta_p} bulunamadı. HPO meta'sı gerekiyor (w,a,b için).")

    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    meta_full = json.loads(meta_full_p.read_text(encoding="utf-8"))

    features_meta = meta_full.get("features", []) or meta.get("features", [])
    best_params = meta.get("best_params", {})
    w = float(best_params.get("w", 0.6))
    a = float(best_params.get("a", 1.0))
    b = float(best_params.get("b", 1.0))

    # Load test data (prefer enriched if available)
    test_cands = [root / "veri" / "test_sessions.parquet", root / "veri" / "enriched_test_data_v2.parquet"]
    test_path = next((p for p in test_cands if p.exists()), None)
    if test_path is None:
        raise FileNotFoundError("Test parquet dosyası bulunamadı (veri/test_sessions.parquet veya enriched_*).")

    test_pl = pl.read_parquet(str(test_path))
    test_df = test_pl.to_pandas()

    features = [f for f in features_meta if f in test_df.columns]
    if not features:
        features = build_features(test_df)

    # Load full models sequentially and predict
    ordered_path = Path(meta_full["model_paths"]["ordered"]) if isinstance(meta_full.get("model_paths"), dict) else (models_dir / "model_ordered_full.cbm")
    clicked_path = Path(meta_full["model_paths"]["clicked"]) if isinstance(meta_full.get("model_paths"), dict) else (models_dir / "model_clicked_full.cbm")

    mo = CatBoostClassifier(); mo.load_model(str(ordered_path))
    p_o = mo.predict_proba(test_df[features])[:, 1]
    del mo; gc.collect()

    mc = CatBoostClassifier(); mc.load_model(str(clicked_path))
    p_c = mc.predict_proba(test_df[features])[:, 1]
    del mc; gc.collect()

    test_df['final_score'] = w * (p_o ** a) + (1.0 - w) * (p_c ** b)

    sub = (
        test_df.sort_values(['session_id', 'final_score'], ascending=[True, False])
        .groupby('session_id')['content_id_hashed']
        .apply(lambda s: ' '.join(s.astype(str)))
        .reset_index()
        .rename(columns={'content_id_hashed': 'prediction'})
    )

    out_path = root / out_name
    sub.to_csv(out_path, index=False)
    print("Submission kaydedildi:", out_path)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default=None, help="Proje kök dizini (varsayılan: bu dosyanın klasörü)")
    p.add_argument("--out", type=str, default="submission_cls_full.csv", help="Çıktı dosya adı")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root = Path(args.root) if args.root else Path(__file__).parent
    main(root, out_name=args.out)
