import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import gc
import pandas as pd
import polars as pl
from catboost import CatBoostClassifier


def _print(msg: str):
    print(f"[submit_single_cls] {msg}")


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
    return [c for c in df.columns if c not in exclude]


def main(root: Path, out_name: str = "submission_single_cls.csv"):
    models_dir = root / "models" / "catboost_cls"
    meta_p = models_dir / "meta_single_cls.json"
    if not meta_p.exists():
        raise FileNotFoundError(f"{meta_p} bulunamadı. Önce train_single_cls_weighted.py ile model eğitin.")

    meta = json.loads(meta_p.read_text(encoding="utf-8"))

    test_cands = [root / "veri" / "test_sessions.parquet", root / "veri" / "enriched_test_data_v2.parquet"]
    test_path = next((p for p in test_cands if p.exists()), None)
    if test_path is None:
        raise FileNotFoundError("Test parquet dosyası bulunamadı (veri/test_sessions.parquet veya enriched_*).")

    test_pl = pl.read_parquet(str(test_path))
    test_df = test_pl.to_pandas()

    features = [f for f in meta.get("features", []) if f in test_df.columns]
    if not features:
        features = build_features(test_df)

    model_path = Path(meta.get("model_path", models_dir / "model_single_weighted.cbm"))
    model = CatBoostClassifier(); model.load_model(str(model_path))

    scores = model.predict_proba(test_df[features])[:, 1]
    del model; gc.collect()

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
    p.add_argument("--out", type=str, default="submission_single_cls.csv", help="Çıktı dosya adı")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root = Path(args.root) if args.root else Path(__file__).parent
    main(root, out_name=args.out)
