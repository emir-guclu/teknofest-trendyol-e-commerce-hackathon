import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import polars as pl
from catboost import CatBoostClassifier

try:
    import optuna
except Exception as e:
    optuna = None

from trendyol_metric_group_auc import score as group_auc_score


warnings.filterwarnings("ignore", category=FutureWarning)


def _print(msg: str):
    print(f"[hpo_optuna_v2] {msg}")


def time_based_val_split(df_pl: pl.DataFrame, val_frac: float = 0.15) -> pd.DataFrame:
    if "ts_hour" in df_pl.columns and df_pl["ts_hour"].dtype != pl.Datetime:
        df_pl = df_pl.with_columns(pl.col("ts_hour").str.to_datetime())
    df_pl = df_pl.sort("ts_hour")
    split_idx = int(len(df_pl) * (1 - val_frac))
    return df_pl[split_idx:].to_pandas()


def build_features(df: pd.DataFrame) -> list[str]:
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


def _find_first(path: Path, names: list[str]) -> Path | None:
    for n in names:
        p = path / n
        if p.exists():
            return p
    return None


def predict_scores(models_dir: Path, df: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Load CatBoost models and return (p_order, p_click). Supports v2/legacy names."""
    mo_p = _find_first(models_dir, ["model_ordered_v2.cbm", "model_ordered.cbm"]) 
    mc_p = _find_first(models_dir, ["model_clicked_v2.cbm", "model_clicked.cbm"]) 
    if mo_p is None or mc_p is None:
        missing = []
        if mo_p is None:
            missing.append("model_ordered_v2.cbm|model_ordered.cbm")
        if mc_p is None:
            missing.append("model_clicked_v2.cbm|model_clicked.cbm")
        raise FileNotFoundError(f"Eksik model dosyaları: {', '.join(missing)} in {models_dir}")

    model_o = CatBoostClassifier(); model_o.load_model(str(mo_p))
    model_c = CatBoostClassifier(); model_c.load_model(str(mc_p))

    # Keep only features existing in df
    feat = [f for f in features if f in df.columns]
    if not feat:
        raise ValueError("Özellik listesi boş. Veri kolonlarını kontrol edin.")

    p_order = model_o.predict_proba(df[feat])[:, 1]
    p_click = model_c.predict_proba(df[feat])[:, 1]
    return p_order, p_click


def local_metric(val_pd: pd.DataFrame, scores: np.ndarray) -> float:
    """Compute competition-like metric using provided trendyol metric implementation."""
    tmp = val_pd.copy()
    tmp["final_score"] = scores

    solution_groups = tmp.groupby('session_id')

    ordered_items = solution_groups.apply(lambda g: ' '.join(g.loc[g['ordered'] == 1, 'content_id_hashed'].astype(str)))
    clicked_items = solution_groups.apply(lambda g: ' '.join(g.loc[g['clicked'] == 1, 'content_id_hashed'].astype(str)))
    all_items = solution_groups['content_id_hashed'].apply(lambda s: ' '.join(s.astype(str)))

    val_solution = pd.DataFrame({
        'ordered_items': ordered_items,
        'clicked_items': clicked_items,
        'all_items': all_items,
    }).reset_index()

    val_submission = (
        tmp.sort_values(['session_id', 'final_score'], ascending=[True, False])
        .groupby('session_id')['content_id_hashed']
        .apply(lambda s: ' '.join(s.astype(str)))
        .reset_index()
        .rename(columns={'content_id_hashed': 'prediction'})
    )
    return float(group_auc_score(val_solution, val_submission, 'session_id'))


def run(root: Path, trials: int = 80, val_frac: float = 0.15, seed: int = 42, write_submission: bool = True):
    if optuna is None:
        raise RuntimeError("optuna yüklü değil. Lütfen 'pip install optuna' ile kurun.")

    models_dir = root / "models"
    # Prefer plain sessions if present; else enriched
    train_cands = [root / "veri/train_sessions.parquet", root / "veri/enriched_train_data_v2.parquet"]
    test_cands = [root / "veri/test_sessions.parquet", root / "veri/enriched_test_data_v2.parquet"]
    train_path = next((p for p in train_cands if p.exists()), None)
    test_path = next((p for p in test_cands if p.exists()), None)
    if train_path is None or test_path is None:
        raise FileNotFoundError("Train/Test parquet dosyaları bulunamadı (veri/train_sessions(.parquet) / veri/enriched_*).")

    _print(f"Using train={train_path}")
    _print(f"Using test={test_path}")

    val_pl = pl.read_parquet(str(train_path))
    test_pl = pl.read_parquet(str(test_path))
    val_pd = time_based_val_split(val_pl, val_frac=val_frac)
    features = build_features(val_pd)

    _print("Predicting probabilities on validation…")
    p_order_val, p_click_val = predict_scores(models_dir, val_pd, features)

    def objective(trial) -> float:
        # w in [0.5, 0.95], exponents in [0.5, 1.5]
        w = trial.suggest_float("w", 0.5, 0.95)
        a = trial.suggest_float("a", 0.5, 1.5)
        b = trial.suggest_float("b", 0.5, 1.5)
        scores = w * (p_order_val ** a) + (1.0 - w) * (p_click_val ** b)
        return local_metric(val_pd, scores)

    _print("Running Optuna study to tune (w, a, b)…")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=trials, show_progress_bar=True)

    best = study.best_trial
    w = best.params["w"]; a = best.params["a"]; b = best.params["b"]
    _print(f"Best params -> w={w:.4f}, a={a:.4f}, b={b:.4f} | local={best.value:.6f}")

    if write_submission:
        _print("Scoring test set and writing submission…")
        test_pd = test_pl.to_pandas()
        # Ensure same features set
        test_features = [f for f in features if f in test_pd.columns]
        p_order_t, p_click_t = predict_scores(models_dir, test_pd, test_features)
        test_pd['final_score'] = w * (p_order_t ** a) + (1.0 - w) * (p_click_t ** b)
        sub = (
            test_pd.sort_values(['session_id', 'final_score'], ascending=[True, False])
            .groupby('session_id')['content_id_hashed']
            .apply(lambda s: ' '.join(s.astype(str)))
            .reset_index()
            .rename(columns={'content_id_hashed': 'prediction'})
        )
        out = root / "submission_hpo_optuna_v2.csv"
        sub.to_csv(out, index=False)
        _print(f"Saved submission: {out}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default=None, help="Proje kök dizini (varsayılan: bu dosyanın klasörü)")
    p.add_argument("--trials", type=int, default=80, help="Optuna deneme sayısı")
    p.add_argument("--val-frac", type=float, default=0.15, help="Validasyon oranı")
    p.add_argument("--no-submit", action="store_true", help="Sadece HPO yap, submission yazma")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root = Path(args.root) if args.root else Path(__file__).parent
    run(root=root, trials=args.trials, val_frac=args.val_frac, write_submission=(not args.no_submit))
