import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
from catboost import CatBoostClassifier
from tqdm import tqdm
from trendyol_metric_group_auc import score

import warnings

# FutureWarning'ları yoksay
warnings.filterwarnings("ignore", category=FutureWarning)


def _print(msg: str):
    print(f"[test_ens_v2] {msg}")


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


def predict_scores(models_dir: Path, df: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    mo_p = models_dir / "model_ordered.cbm"
    mc_p = models_dir / "model_clicked.cbm"
    if not mo_p.exists() or not mc_p.exists():
        raise FileNotFoundError(f"Eksik model dosyası: {mo_p if not mo_p.exists() else mc_p}")
    model_o = CatBoostClassifier(); model_o.load_model(str(mo_p))
    model_c = CatBoostClassifier(); model_c.load_model(str(mc_p))
    p_order = model_o.predict_proba(df[features])[:, 1]
    p_click = model_c.predict_proba(df[features])[:, 1]
    return p_order, p_click


# YENİ KOD (UYARIYI GİDEREN)
def local_metric(val_pd: pd.DataFrame, scores: np.ndarray) -> float:
    tmp = val_pd.copy()
    tmp["final_score"] = scores
    
    # Gruplama işlemini bir kere yapıp değişkene atayalım
    solution_groups = tmp.groupby('session_id')
    
    # Her bir işlemi ayrı ayrı ve daha açık şekilde yapalım
    ordered_items = solution_groups.apply(lambda g: ' '.join(g.loc[g['ordered'] == 1, 'content_id_hashed'].astype(str)))
    clicked_items = solution_groups.apply(lambda g: ' '.join(g.loc[g['clicked'] == 1, 'content_id_hashed'].astype(str)))
    all_items = solution_groups['content_id_hashed'].apply(lambda s: ' '.join(s.astype(str))) # Bu daha da verimli
    
    # Ayrı ayrı hesaplanan Series'leri tek bir DataFrame'de birleştirelim
    val_solution = pd.DataFrame({
        'ordered_items': ordered_items,
        'clicked_items': clicked_items,
        'all_items': all_items
    }).reset_index()

    val_submission = (
        tmp.sort_values(['session_id', 'final_score'], ascending=[True, False])
        .groupby('session_id')['content_id_hashed']
        .apply(lambda s: ' '.join(s.astype(str)))
        .reset_index()
        .rename(columns={'content_id_hashed': 'prediction'})
    )
    return score(val_solution, val_submission, 'session_id')


def run(root: Path, w: float | None):
    models_dir = root / "models"
    train_v2_plus = root / "veri/train_sessions.parquet"
    test_v2_plus = root / "veri/test_sessions.parquet"
    
    train_v2 = root / "veri/enriched_train_data_v2.parquet"
    test_v2 = root / "veri/enriched_test_data_v2.parquet"
    val_pl = pl.read_parquet(str(train_v2_plus if train_v2_plus.exists() else train_v2))
    test_pl = pl.read_parquet(str(test_v2_plus if test_v2_plus.exists() else test_v2))
    val_pd = time_based_val_split(val_pl, val_frac=0.15)
    features = build_features(val_pd)

    _print("Predicting on validation…")
    p_order_val, p_click_val = predict_scores(models_dir, val_pd, features)

        # <<< DEĞİŞİKLİK BAŞLANGICI >>>
    _print("Finding best ensemble weight w by searching over validation set...")
    
    best_s = 0.0
    best_w = 0.7 # Varsayılan değer
    
    # Ağırlık arama uzayı (daha hassas arama için adım sayısını artırabilirsiniz)
    search_space = np.linspace(0.5, 0.9, 41)
    
    for w_candidate in tqdm(search_space, desc="Searching for best w"):
        scores = w_candidate * p_order_val + (1.0 - w_candidate) * p_click_val
        s_candidate = local_metric(val_pd, scores)
        # print(f"  -> w={w_candidate:.3f} | Local Score={s_candidate:.5f}") # İsteğe bağlı: her adımı yazdırmak için
        if s_candidate > best_s:
            best_s = s_candidate
            best_w = w_candidate

    _print(f"Optimal weight found! w = {best_w:.3f} with local score = {best_s:.5f} FOR V2")
    
    # En iyi bulunan ağırlığı kullan
    w = best_w
    # <<< DEĞİŞİKLİK SONU >>>

    _print("Scoring test and writing submission…")
    test_pd = test_pl.to_pandas()
    p_order_t, p_click_t = predict_scores(models_dir, test_pd, features)
    test_pd['final_score'] = w * p_order_t + (1.0 - w) * p_click_t
    sub = (
        test_pd.sort_values(['session_id', 'final_score'], ascending=[True, False])
        .groupby('session_id')['content_id_hashed']
        .apply(lambda s: ' '.join(s.astype(str)))
        .reset_index()
        .rename(columns={'content_id_hashed': 'prediction'})
    )
    out = root / "submission_catboost_ens_v2.csv"
    sub.to_csv(out, index=False)
    _print(f"Saved submission: {out}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--w", type=float, default=0.70, help="Weight for ordered proba; default 0.70 (no grid-search)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root = Path("C:/Projects/Veri/YeniDeneme")
    run(root, args.w)
