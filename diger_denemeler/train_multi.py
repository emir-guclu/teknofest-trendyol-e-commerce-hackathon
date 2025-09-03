import os
import sys
import json
from pathlib import Path
import polars as pl
import pandas as pd
from catboost import CatBoostClassifier

def _print(msg: str):
    print(f"[train_multi] {msg}")

def load_cat_features_list(root: Path) -> list[str]:
    p = root / "cat_features_v2.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return []

def main():
    if len(sys.argv) < 2:
        print("Kullanım: python train_multi.py <veri_seti_no>")
        sys.exit(1)
    veri_seti_no = sys.argv[1]
    assert veri_seti_no in ("1", "2", "3", "4"), "veri_seti_no 1, 2, 3 veya 4 olmalı!"

    root = Path("C:/Users/pc/Desktop/trendyol_hekaton/YeniDeneme")
    train_path = root / f"veri/enriched_train_data_{veri_seti_no}.parquet"
    out_dir = root / "models"
    out_dir.mkdir(parents=True, exist_ok=True)

    _print(f"Loading train parquet (set {veri_seti_no})…")
    df_pl = pl.read_parquet(str(train_path))
    if "ts_hour" in df_pl.columns and df_pl["ts_hour"].dtype != pl.Datetime:
        df_pl = df_pl.with_columns(pl.col("ts_hour").str.to_datetime())
    df_pl = df_pl.sort("ts_hour")
    split_idx = int(len(df_pl) * 0.85)
    train_pl, val_pl = df_pl[:split_idx], df_pl[split_idx:]
    train_pd = train_pl.to_pandas()
    val_pd = val_pl.to_pandas()

    # Targets and features
    targets = ["ordered", "clicked"]
    exclude_cols = set(targets + [
        "ts_hour",
        "session_id",
        "content_creation_date",
        "update_date",
        "added_to_cart",
        "added_to_fav",
    ])
    features = [c for c in train_pd.columns if c not in exclude_cols]

    # Cat features from file if available
    cat_features_list = load_cat_features_list(root)
    if not cat_features_list:
        for c in features:
            if str(train_pd[c].dtype) in ("object", "string", "string[python]"):
                cat_features_list.append(c)

    # Eğitimdeki feature ve cat_features listesini kaydet
    with open(root / f"veri/features_{veri_seti_no}.json", "w", encoding="utf-8") as f:
        json.dump({"features": features, "cat_features": cat_features_list}, f, ensure_ascii=False, indent=2)

    _print(f"n_features={len(features)}, n_cats={len(cat_features_list)}")

    # Split targets
    X_train, X_val = train_pd[features], val_pd[features]
    X_train = X_train.fillna(-999)
    X_val = X_val.fillna(-999)
    y_order_train, y_order_val = train_pd["ordered"], val_pd["ordered"]
    y_click_train, y_click_val = train_pd["clicked"], val_pd["clicked"]

    # Base params
    base_params = dict(
        iterations=1000,
        eval_metric="AUC",
        task_type="GPU",
        random_seed=42,
        verbose=100,
        early_stopping_rounds=100,
    )

    # ordered model
    _print("Training CatBoost for 'ordered'…")
    order_pos = max(1, int(y_order_train.sum()))
    order_neg = max(1, int((y_order_train == 0).sum()))
    order_spw = order_neg / order_pos
    params_ordered = base_params | {"scale_pos_weight": order_spw}
    model_ordered = CatBoostClassifier(**params_ordered, cat_features=cat_features_list)
    model_ordered.fit(X_train, y_order_train, eval_set=(X_val, y_order_val))
    ordered_path = out_dir / f"model_ordered_{veri_seti_no}.cbm"
    model_ordered.save_model(str(ordered_path))
    _print(f"Saved: {ordered_path}")

    # clicked model
    _print("Training CatBoost for 'clicked'…")
    click_pos = max(1, int(y_click_train.sum()))
    click_neg = max(1, int((y_click_train == 0).sum()))
    click_spw = click_neg / click_pos
    params_clicked = base_params | {"scale_pos_weight": click_spw}
    model_clicked = CatBoostClassifier(**params_clicked, cat_features=cat_features_list)
    model_clicked.fit(X_train, y_click_train, eval_set=(X_val, y_click_val))
    clicked_path = out_dir / f"model_clicked_{veri_seti_no}.cbm"
    model_clicked.save_model(str(clicked_path))
    _print(f"Saved: {clicked_path}")

    _print("Done.")

if __name__ == "__main__":
    main()
