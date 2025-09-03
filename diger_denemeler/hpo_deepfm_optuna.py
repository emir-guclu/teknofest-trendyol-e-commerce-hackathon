import argparse
import json
import os
from pathlib import Path

import numpy as np
import optuna
import polars as pl
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from trendyol_metric_group_auc import score as group_auc_score
from train_deepfm_weighted import (
    DeepFM,
    DeepFMDataset,
    build_features,
    time_based_split,
    collate,
    build_val_frames,
)


def _print(msg: str):
    print(f"[deepfm-hpo] {msg}")


def objective(trial: optuna.Trial, root: Path, val_frac: float, w_order: float, w_click: float, epochs: int) -> float:
    # Data load and split
    train_cands = [root / "veri" / "train_sessions.parquet", root / "veri" / "enriched_train_data_v2.parquet"]
    train_path = next((p for p in train_cands if p.exists()), None)
    if train_path is None:
        raise FileNotFoundError("Train parquet not found.")
    df_pl = pl.read_parquet(str(train_path))
    train_pd, val_pd = time_based_split(df_pl, val_frac=val_frac)

    # Filter both_zero from training only
    both_zero_mask = (train_pd["ordered"] == 0) & (train_pd["clicked"] == 0)
    train_pd = train_pd.loc[~both_zero_mask].reset_index(drop=True)

    # Features
    cat_cols, num_cols = build_features(train_pd)

    # Hyperparams
    emb_dim = trial.suggest_categorical("emb_dim", [8, 16, 24, 32])
    base = trial.suggest_categorical("hidden_base", [128, 256, 384])
    depth = trial.suggest_int("depth", 2, 3)
    hidden_dims = [base, base // 2] if depth == 2 else [base, base // 2, max(32, base // 4)]
    dropout = trial.suggest_float("dropout", 0.0, 0.4)
    lr = trial.suggest_float("lr", 3e-4, 3e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32768, 65536, 131072])
    bucket = trial.suggest_categorical("cat_bucket_size", [20000, 50000, 100000])
    cat_bucket_sizes = {c: bucket for c in cat_cols}

    # Stats
    num_means = {c: float(train_pd[c].mean()) for c in num_cols}
    num_stds = {c: float(train_pd[c].std(ddof=0) or 1.0) for c in num_cols}

    # Datasets & loaders (Windows-safe)
    ds_tr = DeepFMDataset(train_pd, cat_cols, num_cols, w_order, w_click, num_means, num_stds, cat_bucket_sizes)
    ds_va = DeepFMDataset(val_pd, cat_cols, num_cols, w_order, w_click, num_means, num_stds, cat_bucket_sizes)
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate, pin_memory=True)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepFM(cat_cols, num_dim=len(num_cols), cat_bucket_sizes=cat_bucket_sizes, emb_dim=emb_dim, hidden_dims=hidden_dims, dropout=dropout).to(device)

    # Train short
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = torch.nn.BCEWithLogitsLoss(reduction='none')
    scaler = GradScaler(enabled=torch.cuda.is_available())

    for _ in range(epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        for cat_idx, num_x, y, w in dl_tr:
            cat_idx = cat_idx.to(device, non_blocking=True)
            num_x = num_x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).view(-1,1)
            w = w.to(device, non_blocking=True).view(-1,1)
            with autocast(enabled=torch.cuda.is_available()):
                logits = model(cat_idx, num_x)
                loss = bce(logits, y)
                loss = (loss * w).mean()
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)

    # Validate metric
    model.eval(); scores=[]
    with torch.no_grad():
        for cat_idx, num_x, _, _ in dl_va:
            with autocast(enabled=torch.cuda.is_available()):
                logits = model(cat_idx.to(device, non_blocking=True), num_x.to(device, non_blocking=True))
                scores.append(torch.sigmoid(logits).squeeze(1).float().cpu().numpy())
    val_scores = np.concatenate(scores) if scores else np.array([])
    sol, sub = build_val_frames(val_pd, val_scores)
    final_score = group_auc_score(sol, sub, "session_id")
    trial.set_user_attr("final_score", final_score)
    return final_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=None)
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--study", type=str, default="deepfm_hpo")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--w_order", type=float, default=0.7)
    ap.add_argument("--w_click", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=1)
    args = ap.parse_args()

    root = Path(args.root) if args.root else Path(__file__).parent
    models_dir = root / "models" / "deepfm"
    models_dir.mkdir(parents=True, exist_ok=True)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=0)
    study = optuna.create_study(direction="maximize", study_name=args.study, sampler=sampler, pruner=pruner)

    def obj(trial):
        return objective(trial, root, args.val_frac, args.w_order, args.w_click, args.epochs)

    _print(f"Starting HPO: trials={args.trials}")
    study.optimize(obj, n_trials=args.trials, show_progress_bar=False)
    _print(f"Best value: {study.best_value:.6f}")
    _print(f"Best params: {study.best_trial.params}")

    best = {
        "best_value": study.best_value,
        "best_params": study.best_trial.params,
        "study_name": args.study,
        "trials": args.trials,
        "val_frac": args.val_frac,
        "weights": {"ordered": args.w_order, "clicked": args.w_click},
    }
    (models_dir / "hpo_deepfm_meta.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    _print(f"Saved best params to {(models_dir / 'hpo_deepfm_meta.json').as_posix()}")


if __name__ == "__main__":
    main()
