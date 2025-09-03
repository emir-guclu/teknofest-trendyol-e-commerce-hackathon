"""
Script to evaluate DeepFM model using Trendyol group AUC metric.

Usage:
  python tools/evaluate_deepfm.py

Requires:
  torch, pandas, numpy
"""
import sys
import json
from pathlib import Path
import warnings
warnings.filterwarnings(
    "ignore",
    message=".*X does not have valid feature names.*",
    category=UserWarning
)
from pathlib import Path
import importlib.machinery
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Load scoring function dynamically using SourceFileLoader
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "trendyol-e-ticaret-hackathonu-2025-kaggle" / "data"
loader = importlib.machinery.SourceFileLoader(
    "trendyol_metric_group_auc", str(DATA_DIR / "trendyol_metric_group_auc.py")
)
tmga = loader.load_module()
score = tmga.score
# Import model and dataset definitions from train_deepfm script
import sys
# Ensure tools directory is in path for train_deepfm import
sys.path.insert(0, str(ROOT / "tools"))
from train_deepfm import SessionDeepFMDataset, DeepFM


def main():
    ROOT = Path(__file__).resolve().parents[1]
    DATA_DIR = ROOT / "trendyol-e-ticaret-hackathonu-2025-kaggle" / "data"
    DP = DATA_DIR / "deepfm"
    # Load encoders and scaler for features
    import pickle
    enc_path = DP / 'cat_encoders.pkl'
    with open(enc_path, 'rb') as f:
        enc_dict = pickle.load(f)
    encoders = enc_dict['encoders']
    cat_cols = enc_dict['cat_cols']
    scaler_path = DP / 'num_scaler.pkl'
    with open(scaler_path, 'rb') as f:
        scaler_dict = pickle.load(f)
    scaler = scaler_dict['scaler']
    num_cols = scaler_dict['num_cols']
    # Load model

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Determine categorical embedding sizes from encoders
    feature_sizes = [len(encoders[c].classes_) for c in cat_cols]
    model = DeepFM(feature_sizes, len(num_cols)).to(device)
    model.load_state_dict(torch.load(DP / 'best_deepfm.pt', map_location=device))
    model.eval()

    # Load processed train sessions for validation
    proc = pd.read_parquet(DATA_DIR / 'processed_train_sessions.parquet')
    # Also load original train for labels
    orig = pd.read_parquet(DATA_DIR / 'train_sessions.parquet')

    submission_rows = []
    solution_rows = []
    # process per session using encoders and scaler
    for session_id, grp in proc.groupby('session_id'):
        # encode categorical features per row
        arr_cat = []
        for col in cat_cols:
            le = encoders[col]
            vals = grp[col].fillna('unknown').astype(str).tolist()
            arr_cat.append(le.transform(vals))
        X_cat = np.stack(arr_cat, axis=1).astype(np.int64)
        # scale numeric features using DataFrame to preserve feature names
        X_num = scaler.transform(grp[num_cols].fillna(0)).astype(np.float32)
        with torch.no_grad():
            x_cat_t = torch.from_numpy(X_cat).to(device)
            x_num_t = torch.from_numpy(X_num).to(device)
            logit1, logit2 = model(x_cat_t, x_num_t)
            # combine scores: weighted sum click*0.3 + order*0.7
            prob = (torch.sigmoid(logit2)*0.7 + torch.sigmoid(logit1)*0.3).cpu().numpy().flatten()
        # rank items
        items = grp['content_id_hashed'].values
        ranked = [x for _,x in sorted(zip(-prob, items), key=lambda z: (z[0], z[1]))]
        submission_rows.append({
            'session_id': session_id,
            'prediction': ' '.join(ranked)
        })
        # solution
        orig_grp = orig[orig['session_id']==session_id]
        all_items = orig_grp['content_id_hashed'].astype(str).tolist()
        ordered_items = orig_grp[orig_grp['ordered']==1]['content_id_hashed'].astype(str).tolist()
        clicked_items = orig_grp[orig_grp['clicked']==1]['content_id_hashed'].astype(str).tolist()
        solution_rows.append({
            'session_id': session_id,
            'all_items': ' '.join(all_items),
            'ordered_items': ' '.join(ordered_items),
            'clicked_items': ' '.join(clicked_items)
        })

    submission_df = pd.DataFrame(submission_rows)
    solution_df = pd.DataFrame(solution_rows)
    # evaluate
    final_score = score(solution_df, submission_df, 'session_id')
    print(f"Final Group AUC Score: {final_score:.4f}")


if __name__=='__main__':
    main()
