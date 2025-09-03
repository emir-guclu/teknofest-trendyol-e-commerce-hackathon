import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple

# Prefer single GPU unless overridden
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import gc
import numpy as np
import pandas as pd
import polars as pl
from catboost import CatBoostRanker, Pool

from trendyol_metric_group_auc import score as group_auc_score
from sklearn.metrics import roc_auc_score


def _print(msg: str):
    print(f"[single_ranker] {msg}")


def build_features(df: pd.DataFrame) -> List[str]:
    targets = ["ordered", "clicked"]
    exclude = set(
        targets
        + [
            "ts_hour",
            "session_id",
            "content_id_hashed",
            "content_creation_date",
            "update_date",
            "added_to_cart",
            "added_to_fav",
        ]
    )
    # Also drop any datetime-like columns and typical time-named fields
    time_name_keywords = (
        "ts_",
        "_ts",
        "date",
        "time",
        "hour",
        "day",
        "week",
        "month",
        "year",
    )
    for c in df.columns:
        dt = df[c].dtype
        if str(dt).startswith("datetime"):
            exclude.add(c)
        else:
            lc = c.lower()
            if any(k in lc for k in time_name_keywords):
                exclude.add(c)
    return [c for c in df.columns if c not in exclude]


def infer_cat_features(df: pd.DataFrame, features: List[str]) -> List[str]:
    cats: List[str] = []
    for c in features:
        dt = str(df[c].dtype)
        if dt in ("object", "string", "string[python]"):
            cats.append(c)
    return cats


def time_based_split(df_pl: pl.DataFrame, val_frac: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if "ts_hour" in df_pl.columns and df_pl["ts_hour"].dtype != pl.Datetime:
        df_pl = df_pl.with_columns(pl.col("ts_hour").str.to_datetime())
    df_pl = df_pl.sort("ts_hour")
    split_idx = int(len(df_pl) * (1 - val_frac))
    return df_pl[:split_idx].to_pandas(), df_pl[split_idx:].to_pandas()


def make_relevance(df: pd.DataFrame, w_order: float, w_click: float) -> np.ndarray:
    # Ensure weights sum to 1 for interpretability
    s = w_order + w_click
    if s <= 0:
        w_order = 0.7
        w_click = 0.3
        s = 1.0
    w_order /= s
    w_click /= s
    ordered = df["ordered"].astype(float).to_numpy()
    clicked = df["clicked"].astype(float).to_numpy()
    rel = w_order * ordered + w_click * clicked
    # In case both are 0/1, values are in {0, w_click, w_order, 1}
    return rel


def build_solution_and_submission(val_df: pd.DataFrame, scores: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
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


def compute_order_clicked_auc(solution: pd.DataFrame, submission: pd.DataFrame, row_id: str = "session_id") -> tuple[float, float]:
    # Build submission dict: session_id -> ordered list of predictions
    sub_map: dict[str, list[str]] = {}
    for _, r in submission.iterrows():
        sid = r[row_id]
        preds = [p.strip() for p in str(r["prediction"]).split() if p.strip()]
        # unique, preserve order
        seen = set()
        uniq = []
        for p in preds:
            if p not in seen:
                seen.add(p)
                uniq.append(p)
        sub_map[sid] = uniq

    ordered_aucs: list[float] = []
    clicked_aucs: list[float] = []
    for _, r in solution.iterrows():
        sid = r[row_id]
        if sid not in sub_map:
            continue
        preds = sub_map[sid]
        ord_items = [x.strip() for x in str(r.get("ordered_items", "")).split() if x.strip()]
        clk_items = [x.strip() for x in str(r.get("clicked_items", "")).split() if x.strip()]

        if preds:
            scores = [len(preds) - i for i in range(len(preds))]
            if ord_items:
                y_true_o = [1 if p in ord_items else 0 for p in preds]
                if len(set(y_true_o)) > 1:
                    try:
                        ordered_aucs.append(float(roc_auc_score(y_true_o, scores)))
                    except Exception:
                        pass
            if clk_items:
                y_true_c = [1 if p in clk_items else 0 for p in preds]
                if len(set(y_true_c)) > 1:
                    try:
                        clicked_aucs.append(float(roc_auc_score(y_true_c, scores)))
                    except Exception:
                        pass

    o_auc = float(np.mean(ordered_aucs)) if ordered_aucs else 0.5
    c_auc = float(np.mean(clicked_aucs)) if clicked_aucs else 0.5
    return o_auc, c_auc


def main(root: Path, w_order: float, w_click: float, val_frac: float, iterations: int, learning_rate: float, depth: int, save_val_csv: bool):
    models_dir = root / "models" / "catboost_ranker"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Load train parquet (prefer the base if present; fall back to enriched)
    train_cands = [root / "veri" / "train_sessions.parquet"]
    train_path = next((p for p in train_cands if p.exists()), None)
    if train_path is None:
        raise FileNotFoundError("Train parquet bulunamadı (veri/train_sessions.parquet veya enriched_*).")

    _print(f"Loading: {train_path}")
    df_pl = pl.read_parquet(str(train_path))
    train_pd, val_pd = time_based_split(df_pl, val_frac=val_frac)

    # Features and cats
    features = build_features(train_pd)
    cat_features = infer_cat_features(train_pd, features)
    _print(f"n_features={len(features)}, n_cat={len(cat_features)} (cats by dtype)")

    # Relevance targets
    y_train = make_relevance(train_pd, w_order=w_order, w_click=w_click)
    y_val = make_relevance(val_pd, w_order=w_order, w_click=w_click)

    # IMPORTANT: CatBoostRanker requires samples to be grouped by group_id contiguously
    # Sort by session_id to ensure grouping, then recompute group codes without sorting
    train_pd = train_pd.sort_values("session_id").reset_index(drop=True)
    val_pd = val_pd.sort_values("session_id").reset_index(drop=True)
    y_train = make_relevance(train_pd, w_order=w_order, w_click=w_click)
    y_val = make_relevance(val_pd, w_order=w_order, w_click=w_click)

    # Group ids (map to integer codes for CatBoost ranker); keep current order
    g_train_codes, _ = pd.factorize(train_pd["session_id"], sort=False)
    g_val_codes, _ = pd.factorize(val_pd["session_id"], sort=False)

    # Build and fit model
    # Build Pools with group information
    train_pool = Pool(
        data=train_pd[features],
        label=y_train,
        group_id=g_train_codes,
        cat_features=cat_features,
    )
    val_pool = Pool(
        data=val_pd[features],
        label=y_val,
        group_id=g_val_codes,
        cat_features=cat_features,
    )

    model = CatBoostRanker(
        iterations=int(iterations),
        learning_rate=float(learning_rate),
        depth=int(depth),
        loss_function="QueryRMSE",
        task_type="GPU",
        devices="0",
        random_seed=42,
        verbose=100,
        early_stopping_rounds=200,
    )
    model.fit(train_pool, eval_set=val_pool)

    # Validation metric (competition metric)
    with np.errstate(all="ignore"):
        val_scores = model.predict(val_pd[features])
    sol, sub = build_solution_and_submission(val_pd, val_scores)
    # Detailed AUCs
    o_auc, c_auc = compute_order_clicked_auc(sol, sub, row_id="session_id")
    local_score = 0.7 * o_auc + 0.3 * c_auc
    _print(f"Validation Ordered AUC: {o_auc:.6f}")
    _print(f"Validation Clicked AUC: {c_auc:.6f}")
    _print(f"Validation Final (0.7*ordered + 0.3*clicked): {local_score:.6f}")

    if save_val_csv:
        (models_dir / "val_solution.csv").write_text(sol.to_csv(index=False), encoding="utf-8")
        (models_dir / "val_submission.csv").write_text(sub.to_csv(index=False), encoding="utf-8")

    # Save model and meta
    out_path = models_dir / "model_ranker_weighted.cbm"
    model.save_model(str(out_path))
    meta = {
        "features": features,
        "cat_features": cat_features,
        "weights": {"ordered": w_order, "clicked": w_click},
        "params": {
            "iterations": int(iterations),
            "learning_rate": float(learning_rate),
            "depth": int(depth),
            "loss_function": "QueryRMSE",
        },
        "val_frac": val_frac,
        "validation": {
            "ordered_auc": o_auc,
            "clicked_auc": c_auc,
            "final": local_score,
        },
        "model_path": out_path.as_posix(),
    }
    (models_dir / "meta_weighted.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _print(f"Saved model to {out_path}")

    # Cleanup
    del model
    gc.collect()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default=None, help="Proje kök dizini; varsayılan bu dosyanın klasörü")
    p.add_argument("--w_order", type=float, default=0.7, help="Ordered ağırlığı (default 0.7)")
    p.add_argument("--w_click", type=float, default=0.3, help="Clicked ağırlığı (default 0.3)")
    p.add_argument("--val_frac", type=float, default=0.15, help="Validation oranı (default 0.15)")
    p.add_argument("--iterations", type=int, default=1500)
    p.add_argument("--learning_rate", type=float, default=0.06)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--save_val_csv", action="store_true", help="Validation solution ve submission CSV'lerini kaydet")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root = Path(args.root) if args.root else Path(__file__).parent
    main(
        root=root,
        w_order=args.w_order,
        w_click=args.w_click,
        val_frac=args.val_frac,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        save_val_csv=args.save_val_csv,
    )
