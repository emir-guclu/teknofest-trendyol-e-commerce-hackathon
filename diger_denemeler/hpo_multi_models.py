import argparse
import json
import logging
from logging.handlers import MemoryHandler
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
import polars as pl

# Pin to a single GPU device if multiple are present (must be set before GPU libs initialize)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from catboost import CatBoostClassifier, CatBoostRanker, Pool

try:
    import xgboost as xgb
except Exception:
    xgb = None

try:
    import optuna
except Exception:
    optuna = None

from trendyol_metric_group_auc import score as group_auc_score


# --------------- Logging ---------------

def setup_logger(log_dir: Path) -> Tuple[logging.Logger, MemoryHandler, Path]:
    """Create a logger that buffers logs and flushes to file at the very end.
    Returns (logger, memory_handler, log_file_path).
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"hpo_multi_{ts}.log"

    logger = logging.getLogger("hpo_multi")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # Console handler (immediate output)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Memory buffer that targets a file, but we flush only at the end
    file_target = logging.FileHandler(str(log_file), encoding="utf-8")
    file_target.setLevel(logging.INFO)

    # Use a very high flushLevel to avoid auto-flush; we'll flush manually
    memh = MemoryHandler(capacity=1024 * 1024, flushLevel=logging.CRITICAL + 1, target=file_target)
    memh.setLevel(logging.INFO)

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    ch.setFormatter(fmt)
    memh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(memh)

    logger.info("Buffered logging initialized; will write file at the end.")
    return logger, memh, log_file


# --------------- Data / Features ---------------

def time_based_splits(df_pl: pl.DataFrame, val_frac: float = 0.15, test_frac: float = 0.05) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if "ts_hour" in df_pl.columns and df_pl["ts_hour"].dtype != pl.Datetime:
        df_pl = df_pl.with_columns(pl.col("ts_hour").str.to_datetime())
    df_pl = df_pl.sort("ts_hour")

    n = len(df_pl)
    test_end = int(n * test_frac)
    val_start = int(n * (1 - val_frac))

    test_pd = df_pl[:test_end].to_pandas()
    val_pd = df_pl[val_start:].to_pandas()
    train_pd = df_pl[test_end:val_start].to_pandas()
    return train_pd, val_pd, test_pd


def _compute_ts_cutoffs_lazy(parquet_path: Path, val_frac: float, test_frac: float) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Read only the ts_hour column to compute time cutoffs for test (head %) and val (tail %).
    This minimizes memory usage versus loading the full table.
    """
    lf = pl.scan_parquet(str(parquet_path)).select(["ts_hour"])  # lazy, only needed column
    df = lf.collect()  # materialize single column
    # Normalize dtype
    if df["ts_hour"].dtype != pl.Datetime:
        # try to parse if string; if numeric, this will raise
        try:
            df = df.with_columns(pl.col("ts_hour").str.to_datetime())
        except Exception:
            df = df.with_columns(pl.col("ts_hour").cast(pl.Datetime))
    df = df.sort("ts_hour")
    n = len(df)
    test_end_idx = max(0, min(n - 1, int(n * test_frac)))
    val_start_idx = max(0, min(n - 1, int(n * (1 - val_frac))))
    ts_test_end = pd.Timestamp(df["ts_hour"][test_end_idx].to_pydatetime())
    ts_val_start = pd.Timestamp(df["ts_hour"][val_start_idx].to_pydatetime())
    return ts_test_end, ts_val_start


def time_based_splits_lazy(parquet_path: Path, needed_columns: List[str], val_frac: float = 0.15, test_frac: float = 0.05) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Memory-friendly split: compute cutoffs from ts_hour only, then load subsets with filters.
    Only needed columns are materialized into memory for each split.
    """
    ts_test_end, ts_val_start = _compute_ts_cutoffs_lazy(parquet_path, val_frac, test_frac)
    # Always ensure ts_hour is present for filtering; add if missing
    cols = list(dict.fromkeys(["ts_hour", *needed_columns]))

    base = pl.scan_parquet(str(parquet_path)).select(cols)
    # Test: ts_hour <= ts_test_end
    test_pd = (
        base.filter(pl.col("ts_hour") <= pl.lit(ts_test_end))
        .collect()
        .to_pandas()
    )
    # Val: ts_hour >= ts_val_start
    val_pd = (
        base.filter(pl.col("ts_hour") >= pl.lit(ts_val_start))
        .collect()
        .to_pandas()
    )
    # Train: between
    train_pd = (
        base.filter((pl.col("ts_hour") > pl.lit(ts_test_end)) & (pl.col("ts_hour") < pl.lit(ts_val_start)))
        .collect()
        .to_pandas()
    )
    return train_pd, val_pd, test_pd


def build_features(df: pd.DataFrame) -> List[str]:
    targets = ["ordered", "clicked"]
    exclude_cols = set(targets + [
        "ts_hour",
        "session_id",
        "content_creation_date",
        "update_date",
        "added_to_cart",
        "added_to_fav",
    ])
    return [c for c in df.columns if c not in exclude_cols]


def infer_cat_features(df: pd.DataFrame, features: List[str]) -> List[str]:
    cats: List[str] = []
    for c in features:
        dt = str(df[c].dtype)
        if dt in ("object", "string", "string[python]"):
            cats.append(c)
    return cats


# --------------- Metric ---------------

def local_metric(df: pd.DataFrame, scores: np.ndarray) -> float:
    tmp = df.copy()
    tmp["final_score"] = scores

    solution_groups = tmp.groupby('session_id', sort=False)
    # include_groups=False to avoid FutureWarning on pandas GroupBy.apply
    ordered_items = solution_groups.apply(
        lambda g: ' '.join(g.loc[g['ordered'] == 1, 'content_id_hashed'].astype(str)), include_groups=False
    )
    clicked_items = solution_groups.apply(
        lambda g: ' '.join(g.loc[g['clicked'] == 1, 'content_id_hashed'].astype(str)), include_groups=False
    )
    all_items = solution_groups['content_id_hashed'].apply(lambda s: ' '.join(s.astype(str)))

    solution = pd.DataFrame({
        'ordered_items': ordered_items,
        'clicked_items': clicked_items,
        'all_items': all_items,
    }).reset_index()

    submission = (
        tmp.sort_values(['session_id', 'final_score'], ascending=[True, False])
        .groupby('session_id')['content_id_hashed']
        .apply(lambda s: ' '.join(s.astype(str)))
        .reset_index()
        .rename(columns={'content_id_hashed': 'prediction'})
    )
    return float(group_auc_score(solution, submission, 'session_id'))


# --------------- XGBoost helpers ---------------

def fit_category_encodings(train: pd.DataFrame, columns: List[str]) -> Dict[str, Dict[str, int]]:
    """Fit category-to-code mappings on train."""
    maps: Dict[str, Dict[str, int]] = {}
    for c in columns:
        if not pd.api.types.is_object_dtype(train[c]) and not pd.api.types.is_string_dtype(train[c]):
            continue
        cats = pd.Series(train[c].astype(str).unique())
        mapping: Dict[str, int] = {str(cat): i for i, cat in enumerate(cats, start=1)}  # 0 reserved for NA/unseen
        maps[c] = mapping
    return maps


def apply_category_encodings(df: pd.DataFrame, enc: Dict[str, Dict[str, int]]) -> pd.DataFrame:
    if not enc:
        return df
    out = df.copy()
    for c, mapping in enc.items():
        s = out[c].astype(str)
        out[c] = s.map(mapping).fillna(0).astype(np.int32)
    return out


# --------------- Result container ---------------

@dataclass
class ModelResult:
    name: str
    val_score: float
    test_score: float
    model_paths: List[Path]
    params: Dict
    extras: Dict


# --------------- Training / HPO ---------------

def hpo_catboost_classifier(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, features: List[str], logger: logging.Logger, trials: int = 40, seed: int = 42) -> ModelResult:
    out_dir = Path(__file__).parent / "models" / "catboost_cls"
    mo_p = out_dir / "model_ordered_best.cbm"
    mc_p = out_dir / "model_clicked_best.cbm"
    meta_p = out_dir / "meta.json"
    if mo_p.exists() and mc_p.exists() and meta_p.exists():
        try:
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            return ModelResult(
                name="catboost_classifier",
                val_score=float(meta.get("val_score", 0.0)),
                test_score=float(meta.get("test_score", 0.0)),
                model_paths=[mo_p, mc_p],
                params=meta.get("best_params", {}),
                extras={"features": meta.get("features", features), "cat_features": meta.get("cat_features", [])},
            )
        except Exception:
            pass  # fallthrough to retrain if meta is corrupted
    cats = infer_cat_features(train, features)

    X_train, X_val = train[features], val[features]
    y_order_tr, y_order_val = train["ordered"], val["ordered"]
    y_click_tr, y_click_val = train["clicked"], val["clicked"]

    order_spw = max(1, int((y_order_tr == 0).sum())) / max(1, int(y_order_tr.sum()))
    click_spw = max(1, int((y_click_tr == 0).sum())) / max(1, int(y_click_tr.sum()))

    def objective(trial):
        params = dict(
            iterations=trial.suggest_int("iterations", 400, 1200),
            depth=trial.suggest_int("depth", 4, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            bootstrap_type="Bernoulli",
            random_seed=seed,
            eval_metric="AUC",
            task_type="GPU",
            verbose=False,
            early_stopping_rounds=100,
        )
        w = trial.suggest_float("w", 0.5, 0.95)
        a = trial.suggest_float("a", 0.5, 1.5)
        b = trial.suggest_float("b", 0.5, 1.5)

        # Train ordered head on GPU, predict, then free
        m_o = CatBoostClassifier(**{**params, "devices": "0", "scale_pos_weight": order_spw, "cat_features": cats})
        m_o.fit(X_train, y_order_tr, eval_set=(X_val, y_order_val))
        p_o = m_o.predict_proba(X_val)[:, 1]
        del m_o
        import gc as _gc
        _gc.collect()

        # Train clicked head on GPU, predict, then free
        m_c = CatBoostClassifier(**{**params, "devices": "0", "scale_pos_weight": click_spw, "cat_features": cats})
        m_c.fit(X_train, y_click_tr, eval_set=(X_val, y_click_val))
        p_c = m_c.predict_proba(X_val)[:, 1]
        del m_c
        _gc.collect()
        scores = w * (p_o ** a) + (1.0 - w) * (p_c ** b)
        return local_metric(val, scores)

    logger.info("[CatBoost-CLS] Starting Optuna HPO…")
    _opt = optuna
    assert _opt is not None, "optuna is required"
    sampler = _opt.samplers.TPESampler(seed=seed)
    study = _opt.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=trials, show_progress_bar=True)

    best = study.best_trial
    logger.info(f"[CatBoost-CLS] Best val score={best.value:.6f}; params={best.params}")

    # Retrain best on train and eval on test
    bp = best.params
    params = dict(
        iterations=bp["iterations"], depth=bp["depth"], learning_rate=bp["learning_rate"],
        l2_leaf_reg=bp["l2_leaf_reg"], subsample=bp["subsample"], random_seed=seed,
    bootstrap_type="Bernoulli", eval_metric="AUC", task_type="GPU", devices="0", verbose=100, early_stopping_rounds=100,
    )
    # Retrain ordered head, predict test, free
    m_o = CatBoostClassifier(**{**params, "devices": "0", "scale_pos_weight": order_spw, "cat_features": cats})
    m_o.fit(X_train, y_order_tr, eval_set=(val[features], y_order_val))
    p_o_t = m_o.predict_proba(test[features])[:, 1]
    del m_o; import gc as _gc; _gc.collect()
    # Retrain clicked head, predict test, free
    m_c = CatBoostClassifier(**{**params, "devices": "0", "scale_pos_weight": click_spw, "cat_features": cats})
    m_c.fit(X_train, y_click_tr, eval_set=(val[features], y_click_val))
    p_c_t = m_c.predict_proba(test[features])[:, 1]
    del m_c; _gc.collect()
    scores_t = bp["w"] * (p_o_t ** bp["a"]) + (1.0 - bp["w"]) * (p_c_t ** bp["b"]) 
    test_score = local_metric(test, scores_t)
    logger.info(f"[CatBoost-CLS] Test score={test_score:.6f}")

    # Save models
    out_dir.mkdir(parents=True, exist_ok=True)
    # Note: models were freed; retrain to save best checkpoints sequentially
    m_o = CatBoostClassifier(**{**params, "devices": "0", "scale_pos_weight": order_spw, "cat_features": cats}); m_o.fit(X_train, y_order_tr)
    mo_p = out_dir / "model_ordered_best.cbm"; m_o.save_model(str(mo_p)); del m_o; _gc.collect()
    m_c = CatBoostClassifier(**{**params, "devices": "0", "scale_pos_weight": click_spw, "cat_features": cats}); m_c.fit(X_train, y_click_tr)
    mc_p = out_dir / "model_clicked_best.cbm"; m_c.save_model(str(mc_p)); del m_c; _gc.collect()
    meta = {
        "type": "catboost_classifier",
        "val_score": best.value,
        "test_score": test_score,
        "best_params": best.params,
        "features": features,
        "cat_features": cats,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return ModelResult(
        name="catboost_classifier",
        val_score=float(best.value or 0.0),
        test_score=float(test_score),
        model_paths=[mo_p, mc_p],
        params=best.params,
        extras={"features": features, "cat_features": cats},
    )


def hpo_catboost_ranker(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, features: List[str], logger: logging.Logger, trials: int = 40, seed: int = 42) -> ModelResult:
    out_dir = Path(__file__).parent / "models" / "catboost_ranker"
    mp = out_dir / "model_ranker_best.cbm"
    meta_p = out_dir / "meta.json"
    if mp.exists() and meta_p.exists():
        try:
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            return ModelResult(
                name="catboost_ranker",
                val_score=float(meta.get("val_score", 0.0)),
                test_score=float(meta.get("test_score", 0.0)),
                model_paths=[mp],
                params=meta.get("best_params", {}),
                extras={"features": meta.get("features", features), "cat_features": meta.get("cat_features", [])},
            )
        except Exception:
            pass  # fallthrough to retrain
    # Ensure groups are contiguous (CatBoost requires grouped queryIds)
    train = train.sort_values([c for c in ["session_id", "ts_hour"] if c in train.columns]).reset_index(drop=True)
    val = val.sort_values([c for c in ["session_id", "ts_hour"] if c in val.columns]).reset_index(drop=True)
    test = test.sort_values([c for c in ["session_id", "ts_hour"] if c in test.columns]).reset_index(drop=True)

    cats = infer_cat_features(train, features)

    def relevance(df: pd.DataFrame) -> np.ndarray:
        # graded relevance: 2 for ordered, 1 for clicked, 0 otherwise
        return (2 * df["ordered"].astype(int) + df["clicked"].astype(int)).to_numpy(dtype=np.float32)

    def group_ids(df: pd.DataFrame) -> np.ndarray:
        return df["session_id"].astype(str).to_numpy()

    def objective(trial):
        params = dict(
            loss_function="YetiRankPairwise",
            iterations=trial.suggest_int("iterations", 400, 1200),
            depth=trial.suggest_int("depth", 4, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
            random_seed=seed,
            task_type="GPU",
            verbose=False,
            early_stopping_rounds=100,
        )
        train_pool = Pool(train[features], label=relevance(train), group_id=group_ids(train), cat_features=cats)
        val_pool = Pool(val[features], label=relevance(val), group_id=group_ids(val), cat_features=cats)

        model = CatBoostRanker(
            loss_function="YetiRankPairwise",
            iterations=params["iterations"],
            depth=params["depth"],
            learning_rate=params["learning_rate"],
            l2_leaf_reg=params["l2_leaf_reg"],
            random_seed=params["random_seed"],
            task_type=params["task_type"],
            devices="0",
            verbose=params["verbose"],
            early_stopping_rounds=params["early_stopping_rounds"],
        )
        model.fit(train_pool, eval_set=val_pool)
        preds = model.predict(val[features])
        del model; import gc as _gc; _gc.collect()
        del train_pool; del val_pool; _gc.collect()
        return local_metric(val, preds)

    logger.info("[CatBoost-RANK] Starting Optuna HPO…")
    _opt = optuna
    assert _opt is not None, "optuna is required"
    sampler = _opt.samplers.TPESampler(seed=seed)
    study = _opt.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=trials, show_progress_bar=True)

    best = study.best_trial
    logger.info(f"[CatBoost-RANK] Best val score={best.value:.6f}; params={best.params}")

    # Retrain and evaluate on test
    params = dict(
        loss_function="YetiRankPairwise",
        iterations=best.params["iterations"], depth=best.params["depth"],
        learning_rate=best.params["learning_rate"], l2_leaf_reg=best.params["l2_leaf_reg"],
    random_seed=seed, task_type="GPU", devices="0", verbose=100, early_stopping_rounds=100,
    )
    tr_pool = Pool(train[features], label=relevance(train), group_id=group_ids(train), cat_features=cats)
    va_pool = Pool(val[features], label=relevance(val), group_id=group_ids(val), cat_features=cats)
    model = CatBoostRanker(
        loss_function="YetiRankPairwise",
        iterations=params["iterations"],
        depth=params["depth"],
        learning_rate=params["learning_rate"],
        l2_leaf_reg=params["l2_leaf_reg"],
        random_seed=params["random_seed"],
        task_type=params["task_type"], devices="0",
        verbose=params["verbose"],
        early_stopping_rounds=params["early_stopping_rounds"],
    )
    model.fit(tr_pool, eval_set=va_pool)

    preds_t = model.predict(test[features])
    test_score = local_metric(test, preds_t)
    logger.info(f"[CatBoost-RANK] Test score={test_score:.6f}")

    out_dir = Path(__file__).parent / "models" / "catboost_ranker"; out_dir.mkdir(parents=True, exist_ok=True)
    mp = out_dir / "model_ranker_best.cbm"
    model.save_model(str(mp))
    meta = {
        "type": "catboost_ranker",
        "val_score": best.value,
        "test_score": test_score,
        "best_params": best.params,
        "features": features,
        "cat_features": cats,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return ModelResult(
        name="catboost_ranker",
        val_score=float(best.value or 0.0),
        test_score=float(test_score),
        model_paths=[mp],
        params=best.params,
        extras={"features": features, "cat_features": cats},
    )


def hpo_xgb_classifier(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, features: List[str], logger: logging.Logger, trials: int = 40, seed: int = 42) -> ModelResult:
    if xgb is None:
        raise RuntimeError("xgboost yüklü değil. 'pip install xgboost' ile kurun.")

    out_dir = Path(__file__).parent / "models" / "xgb_cls"
    mo_p = out_dir / "model_ordered_best.json"
    mc_p = out_dir / "model_clicked_best.json"
    meta_p = out_dir / "meta.json"
    if mo_p.exists() and mc_p.exists() and meta_p.exists():
        try:
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            return ModelResult(
                name="xgb_classifier",
                val_score=float(meta.get("val_score", 0.0)),
                test_score=float(meta.get("test_score", 0.0)),
                model_paths=[mo_p, mc_p],
                params=meta.get("best_params", {}),
                extras={"features": meta.get("features", features), "enc_cats": meta.get("enc_cats", [])},
            )
        except Exception:
            pass  # fallthrough to retrain

    cats = [c for c in features if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
    enc = fit_category_encodings(train, cats)

    X_tr = apply_category_encodings(train[features], enc)
    X_va = apply_category_encodings(val[features], enc)
    X_te = apply_category_encodings(test[features], enc)

    y_order_tr, y_order_va = train["ordered"], val["ordered"]
    y_click_tr, y_click_va = train["clicked"], val["clicked"]

    def objective(trial):
        params = dict(
            max_depth=trial.suggest_int("max_depth", 4, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            n_estimators=trial.suggest_int("n_estimators", 300, 1500),
            objective="binary:logistic",
            random_state=seed,
            tree_method="gpu_hist",
            predictor="gpu_predictor",
            eval_metric="auc",
        )
        w = trial.suggest_float("w", 0.5, 0.95)
        a = trial.suggest_float("a", 0.5, 1.5)
        b = trial.suggest_float("b", 0.5, 1.5)
        xgb_mod = cast(Any, xgb)
        # Train ordered head on GPU
        mo = xgb_mod.XGBClassifier(**params)
        mo.fit(X_tr, y_order_tr, eval_set=[(X_va, y_order_va)], verbose=False)
        p_o = mo.predict_proba(X_va)[:, 1]
        del mo; import gc as _gc; _gc.collect()
        # Train clicked head on GPU
        mc = xgb_mod.XGBClassifier(**params)
        mc.fit(X_tr, y_click_tr, eval_set=[(X_va, y_click_va)], verbose=False)
        p_c = mc.predict_proba(X_va)[:, 1]
        del mc; _gc.collect()
        scores = w * (p_o ** a) + (1.0 - w) * (p_c ** b)
        return local_metric(val, scores)

    logger.info("[XGB-CLS] Starting Optuna HPO…")
    _opt = optuna
    assert _opt is not None, "optuna is required"
    sampler = _opt.samplers.TPESampler(seed=seed)
    study = _opt.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=trials, show_progress_bar=True)

    best = study.best_trial
    logger.info(f"[XGB-CLS] Best val score={best.value:.6f}; params={best.params}")

    # Retrain and evaluate on test
    bp = best.params
    params = dict(
        max_depth=bp["max_depth"], learning_rate=bp["learning_rate"], subsample=bp["subsample"],
        colsample_bytree=bp["colsample_bytree"], reg_lambda=bp["reg_lambda"], reg_alpha=bp["reg_alpha"],
        n_estimators=bp["n_estimators"], objective="binary:logistic", random_state=seed,
        tree_method="gpu_hist", predictor="gpu_predictor", eval_metric="auc",
    )
    xgb_mod = cast(Any, xgb)
    # Train ordered head -> predict -> save -> free
    mo = xgb_mod.XGBClassifier(**params)
    mo.fit(X_tr, y_order_tr, eval_set=[(X_va, y_order_va)], verbose=False)
    p_o_t = mo.predict_proba(X_te)[:, 1]
    out_dir.mkdir(parents=True, exist_ok=True)
    mo_p = out_dir / "model_ordered_best.json"
    mo.save_model(str(mo_p))
    del mo; import gc as _gc; _gc.collect()
    # Train clicked head -> predict -> save -> free
    mc = xgb_mod.XGBClassifier(**params)
    mc.fit(X_tr, y_click_tr, eval_set=[(X_va, y_click_va)], verbose=False)
    p_c_t = mc.predict_proba(X_te)[:, 1]
    mc_p = out_dir / "model_clicked_best.json"
    mc.save_model(str(mc_p))
    del mc; _gc.collect()
    scores_t = bp["w"] * (p_o_t ** bp["a"]) + (1.0 - bp["w"]) * (p_c_t ** bp["b"]) 
    test_score = local_metric(test, scores_t)
    logger.info(f"[XGB-CLS] Test score={test_score:.6f}")
    meta = {
        "type": "xgb_classifier",
        "val_score": best.value,
        "test_score": test_score,
        "best_params": best.params,
        "features": features,
        "enc_cats": list(enc.keys()),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return ModelResult(
        name="xgb_classifier",
        val_score=float(best.value or 0.0),
        test_score=float(test_score),
        model_paths=[mo_p, mc_p],
        params=best.params,
        extras={"features": features, "enc_cats": enc},
    )


def hpo_xgb_ranker(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, features: List[str], logger: logging.Logger, trials: int = 40, seed: int = 42) -> ModelResult:
    if xgb is None:
        raise RuntimeError("xgboost yüklü değil. 'pip install xgboost' ile kurun.")

    out_dir = Path(__file__).parent / "models" / "xgb_ranker"
    mp = out_dir / "model_ranker_best.json"
    meta_p = out_dir / "meta.json"
    if mp.exists() and meta_p.exists():
        try:
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            return ModelResult(
                name="xgb_ranker",
                val_score=float(meta.get("val_score", 0.0)),
                test_score=float(meta.get("test_score", 0.0)),
                model_paths=[mp],
                params=meta.get("best_params", {}),
                extras={"features": meta.get("features", features), "enc_cats": meta.get("enc_cats", [])},
            )
        except Exception:
            pass  # fallthrough to retrain

    # Ensure contiguous grouping by session for XGBRanker
    train_r = train.sort_values(["session_id"]).reset_index(drop=True)
    val_r = val.sort_values(["session_id"]).reset_index(drop=True)
    test_r = test.sort_values(["session_id"]).reset_index(drop=True)

    cats = [c for c in features if pd.api.types.is_object_dtype(train_r[c]) or pd.api.types.is_string_dtype(train_r[c])]
    enc = fit_category_encodings(train_r, cats)

    def rel(df: pd.DataFrame) -> np.ndarray:
        return (2 * df["ordered"].astype(int) + df["clicked"].astype(int)).to_numpy(dtype=np.float32)

    def groups(df: pd.DataFrame) -> List[int]:
        # group sizes per session in current ordering
        return df.groupby('session_id').size().to_list()

    X_tr = apply_category_encodings(train_r[features], enc)
    X_va = apply_category_encodings(val_r[features], enc)
    X_te = apply_category_encodings(test_r[features], enc)

    y_tr = rel(train_r); y_va = rel(val_r); y_te = rel(test_r)

    def objective(trial):
        params = dict(
            max_depth=trial.suggest_int("max_depth", 4, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            n_estimators=trial.suggest_int("n_estimators", 300, 1500),
            objective="rank:pairwise",
            random_state=seed,
            tree_method="gpu_hist",
            predictor="gpu_predictor",
        )
        xgb_mod = cast(Any, xgb)
        model = xgb_mod.XGBRanker(**params)
        model.fit(X_tr, y_tr, group=groups(train_r), eval_set=[(X_va, y_va)], eval_group=[groups(val_r)], verbose=False)
        preds = model.predict(X_va)
        del model; import gc as _gc; _gc.collect()
        return local_metric(val_r, preds)

    logger.info("[XGB-RANK] Starting Optuna HPO…")
    _opt = optuna
    assert _opt is not None, "optuna is required"
    sampler = _opt.samplers.TPESampler(seed=seed)
    study = _opt.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=trials, show_progress_bar=True)

    best = study.best_trial
    logger.info(f"[XGB-RANK] Best val score={best.value:.6f}; params={best.params}")

    bp = best.params
    params = dict(
        max_depth=bp["max_depth"], learning_rate=bp["learning_rate"], subsample=bp["subsample"],
        colsample_bytree=bp["colsample_bytree"], reg_lambda=bp["reg_lambda"], reg_alpha=bp["reg_alpha"],
        n_estimators=bp["n_estimators"], objective="rank:pairwise", random_state=seed,
        tree_method="gpu_hist", predictor="gpu_predictor",
    )
    xgb_mod = cast(Any, xgb)
    model = xgb_mod.XGBRanker(**params)
    model.fit(X_tr, y_tr, group=groups(train_r), eval_set=[(X_va, y_va)], eval_group=[groups(val_r)], verbose=False)
    preds_t = model.predict(X_te)
    test_score = local_metric(test_r, preds_t)
    logger.info(f"[XGB-RANK] Test score={test_score:.6f}")

    out_dir = Path(__file__).parent / "models" / "xgb_ranker"; out_dir.mkdir(parents=True, exist_ok=True)
    mp = out_dir / "model_ranker_best.json"
    model.save_model(str(mp))
    del model; import gc as _gc; _gc.collect()
    meta = {
        "type": "xgb_ranker",
        "val_score": best.value,
        "test_score": test_score,
        "best_params": best.params,
        "features": features,
        "enc_cats": list(enc.keys()),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return ModelResult(
        name="xgb_ranker",
        val_score=float(best.value or 0.0),
        test_score=float(test_score),
        model_paths=[mp],
        params=best.params,
        extras={"features": features, "enc_cats": enc},
    )


# --------------- Runner ---------------

def run(root: Path, trials: int = 40, val_frac: float = 0.15, test_frac: float = 0.05, seed: int = 42):
    if optuna is None:
        raise RuntimeError("optuna yüklü değil. 'pip install optuna' ile kurun.")

    logger, memh, log_file = setup_logger(root / "logs")
    try:
        models_dir = root / "models"; models_dir.mkdir(parents=True, exist_ok=True)

        # Data
        train_cands = [root / "veri/train_sessions.parquet", root / "veri/enriched_train_data_v2.parquet"]
        test_cands = [root / "veri/test_sessions.parquet", root / "veri/enriched_test_data_v2.parquet"]
        train_path = next((p for p in train_cands if p.exists()), None)
        test_path = next((p for p in test_cands if p.exists()), None)
        if train_path is None or test_path is None:
            raise FileNotFoundError("Train/Test parquet dosyaları bulunamadı (veri/train_sessions(.parquet) / veri/enriched_*).")

        logger.info(f"Using train={train_path}")
        logger.info(f"Using test={test_path}")

        df_pl = pl.read_parquet(str(train_path))
        tr_pd, va_pd, te_pd = time_based_splits(df_pl, val_frac=val_frac, test_frac=test_frac)
        features = build_features(tr_pd)
        logger.info(f"n_train={len(tr_pd)}, n_val={len(va_pd)}, n_test={len(te_pd)}, n_features={len(features)}")

        results: List[ModelResult] = []

        # 1) CatBoost Classifier (dual heads + blend)
        results.append(hpo_catboost_classifier(tr_pd, va_pd, te_pd, features, logger, trials=trials, seed=seed))

        # 2) CatBoost Ranker
        results.append(hpo_catboost_ranker(tr_pd, va_pd, te_pd, features, logger, trials=trials, seed=seed))

        # 3) XGBoost Classifier
        results.append(hpo_xgb_classifier(tr_pd, va_pd, te_pd, features, logger, trials=trials, seed=seed))

        # 4) XGBoost Ranker
        results.append(hpo_xgb_ranker(tr_pd, va_pd, te_pd, features, logger, trials=trials, seed=seed))

        # Pick best by test score
        best_overall = max(results, key=lambda r: r.test_score)
        logger.info("Best overall model: %s | test=%.6f val=%.6f", best_overall.name, best_overall.test_score, best_overall.val_score)

        # Create submission using best model
        # Load raw test parquet and score
        test_pl = pl.read_parquet(str(test_path))
        test_pd_full = test_pl.to_pandas()
        features_full = [f for f in features if f in test_pd_full.columns]

        if best_overall.name == "catboost_classifier":
            # reload ordered head -> predict -> free
            mo = CatBoostClassifier(); mo.load_model(str(best_overall.model_paths[0]))
            w = best_overall.params["w"]; a = best_overall.params["a"]; b = best_overall.params["b"]
            p_o = mo.predict_proba(test_pd_full[features_full])[:, 1]
            del mo; import gc as _gc; _gc.collect()
            # reload clicked head -> predict -> free
            mc = CatBoostClassifier(); mc.load_model(str(best_overall.model_paths[1]))
            p_c = mc.predict_proba(test_pd_full[features_full])[:, 1]
            del mc; _gc.collect()
            test_pd_full['final_score'] = w * (p_o ** a) + (1.0 - w) * (p_c ** b)
        elif best_overall.name == "catboost_ranker":
            mr = CatBoostRanker(); mr.load_model(str(best_overall.model_paths[0]))
            test_pd_full['final_score'] = mr.predict(test_pd_full[features_full])
        elif best_overall.name == "xgb_classifier":
            if xgb is None:
                raise RuntimeError("xgboost missing at submission time.")
            enc = best_overall.extras.get("enc_cats", {})
            X_full = apply_category_encodings(test_pd_full[features_full], enc)
            w = best_overall.params["w"]; a = best_overall.params["a"]; b = best_overall.params["b"]
            mo = xgb.XGBClassifier(); mo.load_model(str(best_overall.model_paths[0]))
            p_o = mo.predict_proba(X_full)[:, 1]
            del mo; import gc as _gc; _gc.collect()
            mc = xgb.XGBClassifier(); mc.load_model(str(best_overall.model_paths[1]))
            p_c = mc.predict_proba(X_full)[:, 1]
            del mc; _gc.collect()
            test_pd_full['final_score'] = w * (p_o ** a) + (1.0 - w) * (p_c ** b)
        else:  # xgb_ranker
            if xgb is None:
                raise RuntimeError("xgboost missing at submission time.")
            mr = xgb.XGBRanker(); mr.load_model(str(best_overall.model_paths[0]))
            enc = best_overall.extras.get("enc_cats", {})
            X_full = apply_category_encodings(test_pd_full[features_full], enc)
            test_pd_full['final_score'] = mr.predict(X_full)

        sub = (
            test_pd_full.sort_values(['session_id', 'final_score'], ascending=[True, False])
            .groupby('session_id')['content_id_hashed']
            .apply(lambda s: ' '.join(s.astype(str)))
            .reset_index()
            .rename(columns={'content_id_hashed': 'prediction'})
        )
        out = root / "submission_best_overall.csv"
        sub.to_csv(out, index=False)
        logger.info(f"Saved submission: {out}")
    finally:
        # Flush buffered logs at the very end
        try:
            memh.flush()
        finally:
            for h in list(logger.handlers):
                try:
                    h.close()
                finally:
                    logger.removeHandler(h)
        print(f"Logs saved to: {log_file}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default=None, help="Proje kök dizini (varsayılan: bu dosyanın klasörü)")
    p.add_argument("--trials", type=int, default=40, help="Optuna deneme sayısı (her model tipi için)")
    p.add_argument("--val-frac", type=float, default=0.15, help="Validasyon oranı (son %)")
    p.add_argument("--test-frac", type=float, default=0.05, help="Test oranı (ilk %)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root = Path(args.root) if args.root else Path(__file__).parent
    run(root=root, trials=args.trials, val_frac=args.val_frac, test_frac=args.test_frac)
