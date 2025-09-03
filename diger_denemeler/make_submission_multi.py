
import pandas as pd
import json
from catboost import CatBoostClassifier
from pathlib import Path

# Dosya yolları
root = Path("C:/Users/pc/Desktop/trendyol_hekaton/YeniDeneme")

model_ordered_path = root / "models/catboost_cls/model_ordered_best.cbm"
model_clicked_path = root / "models/model_clicked_4.cbm"
test_path = root / "veri/enriched_test_data_4.parquet"  # GERÇEK TEST SETİ
features_path = root / "veri/features_4.json"
submission_path = root / "submission_model_4.csv"

# Eğitimdeki feature ve cat_features listesini oku
with open(features_path, "r", encoding="utf-8") as f:
    feature_info = json.load(f)
features = feature_info["features"]
cat_features = feature_info["cat_features"]


# Test verisini oku ve feature sırasını eğitimle aynı tut
X_test_full = pd.read_parquet(test_path)
X_test = X_test_full[features].fillna(-999)

# Ordered model
model_ordered = CatBoostClassifier()
model_ordered.load_model(str(model_ordered_path))
ordered_pred = model_ordered.predict_proba(X_test)[:, 1]

# Clicked model
model_clicked = CatBoostClassifier()
model_clicked.load_model(str(model_clicked_path))
clicked_pred = model_clicked.predict_proba(X_test)[:, 1]

# Submission formatı

submission = pd.DataFrame({
    "session_id": X_test_full["session_id"],
    "ordered_score": ordered_pred,
    "clicked_score": clicked_pred
})
submission.to_csv(submission_path, index=False)
print(f"Submission dosyası oluşturuldu: {submission_path}")
