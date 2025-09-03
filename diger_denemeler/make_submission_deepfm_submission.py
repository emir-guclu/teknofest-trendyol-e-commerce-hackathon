"""
Script to generate submission CSV for DeepFM model.
Usage:
  python tools/make_submission_deepfm_submission.py
"""
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import warnings
from torch.cuda.amp import autocast

# suppress warnings for feature name mismatch
warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)

# Import model class from training script
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from train_deepfm import DeepFM


def main():
    ROOT = Path(__file__).resolve().parents[1]
    DATA_DIR = ROOT / "trendyol-e-ticaret-hackathonu-2025-kaggle" / "data"
    # Load processed test sessions
    proc_test = pd.read_parquet(DATA_DIR / "processed_test_sessions.parquet")
    # Load encoders and scaler
    with open(DATA_DIR / "deepfm" / "cat_encoders.pkl", 'rb') as f:
        enc_data = pickle.load(f)
    encoders = enc_data['encoders']
    cat_cols = enc_data['cat_cols']
    with open(DATA_DIR / "deepfm" / "num_scaler.pkl", 'rb') as f:
        scaler_data = pickle.load(f)
    scaler = scaler_data['scaler']
    num_cols = scaler_data['num_cols']
    # Load model meta for feature sizes
    meta = json.loads((DATA_DIR / "deepfm" / "meta_deepfm.json").read_text())
    feature_sizes = meta['feature_sizes'][:len(cat_cols)]
    # Load trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepFM(feature_sizes, len(num_cols), embed_dim=16, hidden_dims=[256,128,64], dropout=0.3).to(device)
    model.load_state_dict(torch.load(DATA_DIR / "deepfm" / "best_deepfm.pt", map_location=device))
    model.eval()

    # Prepare features arrays
    # Categorical
    cat_arrays = []
    for col in cat_cols:
        le = encoders[col]
        vals = proc_test[col].fillna('unknown').astype(str)
        cat_arrays.append(le.transform(vals.tolist()))
    X_cat = np.vstack(cat_arrays).T.astype(np.int64) if cat_cols else np.zeros((len(proc_test),0), dtype=np.int64)
    # Numeric
    X_num = scaler.transform(proc_test[num_cols].fillna(0).values).astype(np.float32) if num_cols else np.zeros((len(proc_test),0), dtype=np.float32)

    # Inference per batch
    scores_click = []
    scores_order = []
    BATCH = 65536
    for i in range(0, len(proc_test), BATCH):
        x_cat_batch = torch.from_numpy(X_cat[i:i+BATCH]).to(device)
        x_num_batch = torch.from_numpy(X_num[i:i+BATCH]).to(device)
        with torch.no_grad():
            with autocast(enabled=torch.cuda.is_available()):
                logit1, logit2 = model(x_cat_batch, x_num_batch)
                p1 = torch.sigmoid(logit1).cpu().numpy().flatten()
                p2 = torch.sigmoid(logit2).cpu().numpy().flatten()
        scores_click.append(p1)
        scores_order.append(p2)
    scores_click = np.concatenate(scores_click)
    scores_order = np.concatenate(scores_order)
    # Combined score
    final_scores = 0.3 * scores_click + 0.7 * scores_order

    # Attach to DataFrame and group by session_id
    # ensure score is float64 to avoid float16 index issues
    proc_test['score'] = final_scores.astype(np.float64)
    submission = (
        proc_test.sort_values(['session_id', 'score'], ascending=[True, False])
        .groupby('session_id')['content_id_hashed']
        .apply(lambda s: ' '.join(s.astype(str)))
        .reset_index()
        .rename(columns={'content_id_hashed': 'prediction'})
    )

    out_path = DATA_DIR / 'submission_deepfm.csv'
    submission.to_csv(out_path, index=False)
    print(f"Submission saved to {out_path}")


if __name__ == '__main__':
    main()
