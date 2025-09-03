from pathlib import Path
import polars as pl
import pandas as pd
import numpy as np
import os
from catboost import CatBoostClassifier
from sympy import root
from trendyol_metric_group_auc import score

print("--- KOD 2: Güvenli Baseline ile Tahmin ve Skorlama ---")

# --- Adım 1: Gerekli Dosyaları Yükleme ---
root = Path("C:/Projects/Veri/YeniDeneme")
VAL_PATH = root / "veri/train_sessions.parquet"
TEST_PATH = root / "veri/test_sessions.parquet"
MODELS_DIR = root / "models"
# Train ile uyumlu: CatBoost model dosya adları (.cbm)
MODEL_ORDERED_PATH = os.path.join(MODELS_DIR, "model_ordered.cbm")
MODEL_CLICKED_PATH = os.path.join(MODELS_DIR, "model_clicked.cbm")

print("Veriler ve kaydedilmiş güvenli modeller yükleniyor...")
val_full_df = pl.read_parquet(VAL_PATH)
test_df_pl = pl.read_parquet(TEST_PATH)
# CatBoost modellerini yükle
model_ordered = CatBoostClassifier()
model_ordered.load_model(MODEL_ORDERED_PATH)
model_clicked = CatBoostClassifier()
model_clicked.load_model(MODEL_CLICKED_PATH)
print("✅ Yükleme tamamlandı.")


# --- Adım 2: Lokal Skor İçin Validation Seti Hazırlama ---
print("\nLokal skor için validation seti hazırlanıyor...")
if "ts_hour" in val_full_df.columns and val_full_df["ts_hour"].dtype != pl.Datetime:
    val_full_df = val_full_df.with_columns(pl.col("ts_hour").str.to_datetime())
val_full_df = val_full_df.sort("ts_hour")
split_index = int(len(val_full_df) * 0.85)
val_pd = val_full_df[split_index:].to_pandas().fillna(0)

targets = ["ordered", "clicked"]
# Train (train_v2.py) ile aynı feature seçimi
exclude_cols = set(targets + [
    "ts_hour",
    "session_id",
    "content_creation_date",
    "update_date",
    "added_to_cart",
    "added_to_fav",
])
features = [c for c in val_pd.columns if c not in exclude_cols]
# Kategorik sütunları kategori tipine çevir (CatBoost uyumlu)
for col in val_pd.columns:
    if val_pd[col].dtype == 'object':
        val_pd[col] = val_pd[col].astype('category')

# CatBoost için pozitif sınıf olasılıkları
val_pd['p_order'] = model_ordered.predict_proba(val_pd[features])[:, 1]
val_pd['p_click'] = model_clicked.predict_proba(val_pd[features])[:, 1]
val_pd['final_score'] = 0.7 * val_pd['p_order'] + 0.3 * val_pd['p_click']
val_solution = val_pd.groupby('session_id').agg(ordered_items=('ordered', lambda x: ' '.join(val_pd.loc[x.index][x == 1]['content_id_hashed'])), clicked_items=('clicked', lambda x: ' '.join(val_pd.loc[x.index][x == 1]['content_id_hashed'])), all_items=('content_id_hashed', ' '.join)).reset_index()
val_submission = val_pd.sort_values(['session_id', 'final_score'], ascending=[True, False]).groupby('session_id')['content_id_hashed'].apply(' '.join).reset_index()
val_submission.rename(columns={'content_id_hashed': 'prediction'}, inplace=True)

try:
    local_final_score = score(val_solution, val_submission, 'session_id')
    print("\n-------------------------------------------")
    print("🏆V1:")
    print(f"🏆 GÜVENLİ BASELINE LOKAL SKORUNUZ: {local_final_score:.5f}")
    print("-------------------------------------------")
except Exception as e:
    print(f"Skor hesaplanırken bir hata oluştu: {e}")


# # --- Adım 3: Kaggle için Submission Dosyası Oluşturma ---
# print("\nTest verisi hazırlanıyor ve Kaggle için submission dosyası oluşturuluyor...")
# test_pd = test_df_pl.to_pandas().fillna(0)
# for col in test_pd.columns:
#     if test_pd[col].dtype == 'object':
#         test_pd[col] = test_pd[col].astype('category')
# p_order_test = model_ordered.predict(test_pd[features], num_iteration=model_ordered.best_iteration)
# p_click_test = model_clicked.predict(test_pd[features], num_iteration=model_clicked.best_iteration)
# test_df_pl = test_df_pl.with_columns(final_score=(0.7 * pl.Series(p_order_test)) + (0.3 * pl.Series(p_click_test)))

# submission_df = test_df_pl.sort(["session_id", "final_score"], descending=True).group_by("session_id").agg(pl.col("content_id_hashed").alias("prediction")).with_columns(pl.col("prediction").list.join(" "))
# expected_rows = 18589
# actual_rows = submission_df.shape[0]
# print(f"\nOluşturulan submission satır sayısı: {actual_rows} (Beklenen: {expected_rows})")
# if actual_rows == expected_rows: print("✅ Satır sayısı doğru.")
# else: print("❌ UYARI: Satır sayısı yanlış!")
# submission_path = "submission.csv"
# submission_df.write_csv(submission_path)
# print(f"\nSubmission dosyası '{submission_path}' olarak kaydedildi.")