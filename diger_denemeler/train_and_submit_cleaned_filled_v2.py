import os
import json
from pathlib import Path
import polars as pl
import pandas as pd
from catboost import CatBoostClassifier

def _print(msg: str):
    print(f"[train_and_submit_cleaned_filled] {msg}")

def load_cat_features_list(root: Path) -> list[str]:
    p = root / "cat_features_v2.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return []


def main():
    root = Path("c:/Users/pc/Desktop/trendyol_hekaton/YD")
    train_path = root / "veri/train_sessions_rerank_filled_final.parquet"
    test_path = root / "veri/test_sessions_rerank_filled_final.parquet"
    out_dir = root / "models"
    out_dir.mkdir(parents=True, exist_ok=True)

    _print("Loading train parquet…")
    df_pl = pl.read_parquet(str(train_path))
    if "ts_hour" in df_pl.columns and df_pl["ts_hour"].dtype != pl.Datetime:
        df_pl = df_pl.with_columns(pl.col("ts_hour").str.to_datetime())
    df_pl = df_pl.sort("ts_hour")
    split_idx = int(len(df_pl) * 0.85)
    train_pl, val_pl = df_pl[:split_idx], df_pl[split_idx:]
    train_pd = train_pl.to_pandas()
    val_pd = val_pl.to_pandas()

    # Targets and features
    targets = ["ordered", "ordered_or_clicked", "clicked"]
    exclude_cols = set(targets + [
        "ts_hour",
        "session_id",
        "content_creation_date",
        "update_date",
        "added_to_cart",
        "added_to_fav",
        "product_text",
        "reranker_score"
    ])
    features = [c for c in train_pd.columns if c not in exclude_cols]

    # Cat features from file if available
    cat_features_list = load_cat_features_list(root)
    if not cat_features_list:
        for c in features:
            if str(train_pd[c].dtype) in ("object", "string", "string[python]"):
                cat_features_list.append(c)

    _print(f"n_features={len(features)}, n_cats={len(cat_features_list)}")

    # Split targets
    X_train, X_val = train_pd[features], val_pd[features]
    y_order_train, y_order_val = train_pd["ordered"], val_pd["ordered"]
    y_ordered_or_clicked_train, y_ordered_or_clicked_val = train_pd["ordered_or_clicked"], val_pd["ordered_or_clicked"]
    y_click_train, y_click_val = train_pd["clicked"], val_pd["clicked"]

    # Base params
    base_params = dict(
        iterations=2000,
        eval_metric="AUC",
        task_type="GPU",
        random_seed=42,
        verbose=100,
        early_stopping_rounds=200,
    )

    # ordered model
    _print("Training CatBoost for 'ordered'…")
    order_pos = max(1, int(y_order_train.sum()))
    order_neg = max(1, int((y_order_train == 0).sum()))
    order_spw = order_neg / order_pos
    params_ordered = base_params | {"scale_pos_weight": order_spw}
    model_ordered = CatBoostClassifier(**params_ordered, cat_features=cat_features_list)
    model_ordered.fit(X_train, y_order_train, eval_set=(X_val, y_order_val))
    ordered_path = out_dir / "model_ordered.cbm"
    model_ordered.save_model(str(ordered_path))
    _print(f"Saved: {ordered_path}")

    # ordered_or_clicked model
    _print("Training CatBoost for 'ordered_or_clicked'…")
    ooc_pos = max(1, int(y_ordered_or_clicked_train.sum()))
    ooc_neg = max(1, int((y_ordered_or_clicked_train == 0).sum()))
    ooc_spw = ooc_neg / ooc_pos
    params_ooc = base_params | {"scale_pos_weight": ooc_spw}
    model_ooc = CatBoostClassifier(**params_ooc, cat_features=cat_features_list)
    model_ooc.fit(X_train, y_ordered_or_clicked_train, eval_set=(X_val, y_ordered_or_clicked_val))
    ooc_path = out_dir / "model_ordered_or_clicked.cbm"
    model_ooc.save_model(str(ooc_path))
    _print(f"Saved: {ooc_path}")

    # clicked model
    _print("Training CatBoost for 'clicked'…")
    click_pos = max(1, int(y_click_train.sum()))
    click_neg = max(1, int((y_click_train == 0).sum()))
    click_spw = click_neg / click_pos
    params_clicked = base_params | {"scale_pos_weight": click_spw}
    model_clicked = CatBoostClassifier(**params_clicked, cat_features=cat_features_list)
    model_clicked.fit(X_train, y_click_train, eval_set=(X_val, y_click_val))
    clicked_path = out_dir / "model_clicked.cbm"
    model_clicked.save_model(str(clicked_path))
    _print(f"Saved: {clicked_path}")

    # Grid search ile en iyi ağırlıkları bul (ordered, ordered_or_clicked, clicked)
    import sys
    sys.path.append(str(root.parent / "trendyol-e-ticaret-hackathonu-2025-kaggle/data"))
    from trendyol_metric_group_auc import score

    val_df = val_pd.copy()
    val_df['p_order'] = model_ordered.predict_proba(val_df[features])[:, 1]
    val_df['p_ooc'] = model_ooc.predict_proba(val_df[features])[:, 1]
    val_df['p_click'] = model_clicked.predict_proba(val_df[features])[:, 1]

    best_score = -1
    best_weights = (0.7, 0.2, 0.1)
    # 0.0-1.0 arası grid search, toplamı 1 olacak şekilde
    for w_order in [i/10 for i in range(3,8)]:
        for w_ooc in [i/10 for i in range(1,8)]:
            w_click = 1.0 - w_order - w_ooc
            if w_click < 0 or w_click > 1:
                continue
            val_df['final_score'] = w_order * val_df['p_order'] + w_ooc * val_df['p_ooc'] + w_click * val_df['p_click']
            # Sıralama: ordered > ordered_or_clicked > clicked > normal
            val_df_sorted = val_df.sort_values(['session_id', 'final_score'], ascending=[True, False])\
                .groupby('session_id')['content_id_hashed'].apply(lambda x: ' '.join(x)).reset_index()
            val_df_sorted.rename(columns={'content_id_hashed': 'prediction'}, inplace=True)
            # ground truth dataframe
            gt_df = val_pd[['session_id', 'ordered', 'clicked', 'ordered_or_clicked', 'content_id_hashed']].copy()
            gt_df['ordered_items'] = gt_df.apply(lambda x: x['content_id_hashed'] if x['ordered'] == 1 else '', axis=1)
            gt_df['clicked_items'] = gt_df.apply(lambda x: x['content_id_hashed'] if x['clicked'] == 1 else '', axis=1)
            gt_df['all_items'] = gt_df['content_id_hashed']
            gt_grouped = gt_df.groupby('session_id').agg({
                'ordered_items': lambda x: ' '.join([i for i in x if i]),
                'clicked_items': lambda x: ' '.join([i for i in x if i]),
                'all_items': lambda x: ' '.join(x)
            }).reset_index()
            try:
                val_score = score(gt_grouped, val_df_sorted, 'session_id')
            except Exception as e:
                continue
            if val_score > best_score:
                best_score = val_score
                best_weights = (w_order, w_ooc, w_click)

    w_order, w_ooc, w_click = best_weights
    _print(f"Best validation metric_group_auc score: {best_score:.4f} with weights ordered={w_order:.2f}, ordered_or_clicked={w_ooc:.2f}, clicked={w_click:.2f}")

    # Submission (sıralama: ordered, ordered_or_clicked, clicked, normal)
    _print("Preparing submission…")
    test_pl = pl.read_parquet(str(test_path))
    test_pd = test_pl.to_pandas()
    for col in test_pd.columns:
        if test_pd[col].dtype == 'object':
            test_pd[col] = test_pd[col].astype('category')
    p_order_test = model_ordered.predict_proba(test_pd[features])[:, 1]
    p_ooc_test = model_ooc.predict_proba(test_pd[features])[:, 1]
    p_click_test = model_clicked.predict_proba(test_pd[features])[:, 1]
    test_pd['final_score'] = (
        w_order * p_order_test +
        w_ooc * p_ooc_test +
        w_click * p_click_test
    )
    submission = test_pd.sort_values(['session_id', 'final_score'], ascending=[True, False])\
        .groupby('session_id')['content_id_hashed'].apply(lambda x: ' '.join(x)).reset_index()
    submission.rename(columns={'content_id_hashed': 'prediction'}, inplace=True)
    submission_path = root / 'submission_with_new_metrics_cleaned_filled_final.csv'
    submission.to_csv(submission_path, index=False)
    _print(f"Submission dosyası kaydedildi: {submission_path}")
    _print("Done.")

if __name__ == "__main__":
    main()
