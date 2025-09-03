import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import pandas as pd
import polars as pl
import torch
import optuna
from optuna.trial import Trial
from torch.utils.data import DataLoader
import sys
import warnings

from train_widedeep_weighted import (
    build_features,
    time_based_split,
    WideDeepDataset,
    WideAndDeep,
    collate,
)
from trendyol_metric_group_auc import score as group_auc_score


def _print(msg: str):
    print(f"[hpo_widedeep] {msg}")


def objective(trial: Trial, root: Path) -> float:
    # Data
    train_cands = [root / "veri" / "train_sessions.parquet", root / "veri" / "enriched_train_data_v2.parquet"]
    train_path = next((p for p in train_cands if p.exists()), None)
    if train_path is None:
        raise FileNotFoundError("Train parquet bulunamadı (veri/train_sessions.parquet veya enriched_*).")

    df_pl = pl.read_parquet(str(train_path))
    train_pd, val_pd = time_based_split(df_pl, val_frac=0.15)

    cat_cols, num_cols = build_features(train_pd)

    # Search space
    emb_dim = trial.suggest_categorical("emb_dim", [8, 16, 24, 32])
    wide_hash_dim = trial.suggest_categorical("wide_hash_dim", [100000, 200000, 400000])
    lr = trial.suggest_float("lr", 5e-4, 5e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16384, 32768, 65536])
    hidden_mult = trial.suggest_categorical("hidden_mult", [0.5, 1.0, 1.5])
    dropout = trial.suggest_float("dropout", 0.05, 0.3)

    hidden_base = [256, 128, 64]
    hidden_dims = [max(32, int(h * hidden_mult)) for h in hidden_base]

    cat_bucket_sizes = {c: 50000 for c in cat_cols}
    num_means = {c: float(train_pd[c].mean()) for c in num_cols}
    num_stds = {c: float(train_pd[c].std(ddof=0) or 1.0) for c in num_cols}

    ds_tr = WideDeepDataset(train_pd, cat_cols, num_cols, 0.7, 0.3, num_means, num_stds, cat_bucket_sizes, wide_hash_dim)
    ds_va = WideDeepDataset(val_pd, cat_cols, num_cols, 0.7, 0.3, num_means, num_stds, cat_bucket_sizes, wide_hash_dim)

    # On Windows, multi-processing DataLoader inside Optuna trials can fail due to spawn/pickle issues.
    # Force single-process DataLoader to avoid OSError: Invalid argument / pickle truncation.
    is_windows = sys.platform.startswith("win")
    if is_windows:
        workers = 0
        dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate, pin_memory=False)
        dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate, pin_memory=False)
    else:
        cpu_cnt = os.cpu_count() or 1
        workers = max(2, min(8, max(1, cpu_cnt // 2)))
        dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=workers, collate_fn=collate, pin_memory=True, persistent_workers=True, prefetch_factor=2)
        dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=workers, collate_fn=collate, pin_memory=True, persistent_workers=True, prefetch_factor=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = WideAndDeep(cat_cols, num_dim=len(num_cols), cat_bucket_sizes=cat_bucket_sizes, wide_hash_dim=wide_hash_dim, emb_dim=emb_dim, hidden_dims=hidden_dims, dropout=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = torch.nn.BCEWithLogitsLoss(reduction='none')
    # Use cuda.amp to avoid type stub issues; silence future deprecation warning
    warnings.filterwarnings("ignore", category=FutureWarning)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_score = -1.0

    for epoch in range(1, 4):  # short epochs for HPO
        model.train()
        opt.zero_grad(set_to_none=True)
        step = 0
        for wide_idx, deep_cat_idx, num_x, y, w in dl_tr:
            wide_idx = wide_idx.to(device, non_blocking=True)
            deep_cat_idx = deep_cat_idx.to(device, non_blocking=True)
            num_x = num_x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).view(-1, 1)
            w = w.to(device, non_blocking=True).view(-1, 1)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(wide_idx, deep_cat_idx, num_x)
                loss = bce(logits, y)
                loss = (loss * w).mean()

            scaler.scale(loss).backward()
            step += 1
            if step % 1 == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

        # Eval
        model.eval()
        with torch.no_grad():
            scores = []
            for wide_idx, deep_cat_idx, num_x, _, _ in dl_va:
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    logits = model(wide_idx.to(device, non_blocking=True), deep_cat_idx.to(device, non_blocking=True), num_x.to(device, non_blocking=True))
                    scores.append(torch.sigmoid(logits).squeeze(1).float().cpu().numpy())
        val_scores = np.concatenate(scores) if scores else np.array([])
        # Build metric frames
        tmp = val_pd.copy(); tmp['final_score'] = val_scores
        g = tmp.groupby('session_id')
        sol = pd.DataFrame({
            'ordered_items': g.apply(lambda grp: ' '.join(grp.loc[grp['ordered']==1, 'content_id_hashed'].astype(str))),
            'clicked_items': g.apply(lambda grp: ' '.join(grp.loc[grp['clicked']==1, 'content_id_hashed'].astype(str))),
            'all_items': g['content_id_hashed'].apply(lambda s: ' '.join(s.astype(str)))
        }).reset_index()
        sub = (
            tmp.sort_values(['session_id','final_score'], ascending=[True,False])
            .groupby('session_id')['content_id_hashed']
            .apply(lambda s: ' '.join(s.astype(str)))
            .reset_index()
            .rename(columns={'content_id_hashed':'prediction'})
        )
        score = float(group_auc_score(sol, sub, 'session_id'))
        if score > best_score:
            best_score = score
        trial.report(score, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_score


def main(root: Path, n_trials: int, study_name: str):
    storage = None  # in-memory
    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=1)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner, study_name=study_name, storage=storage)
    study.optimize(lambda t: objective(t, root), n_trials=n_trials, show_progress_bar=True)

    print("Best Value:", study.best_value)
    print("Best Params:")
    for k, v in study.best_trial.params.items():
        print(f"  - {k}: {v}")

    # Save
    out = {
        "best_value": study.best_value,
        "best_params": study.best_trial.params,
    }
    (root / 'models' / 'widedeep' / 'hpo_widedeep_meta.json').write_text(json.dumps(out, indent=2), encoding='utf-8')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default=None)
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--study", type=str, default="widedeep_hpo")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root = Path(args.root) if args.root else Path(__file__).parent
    main(root, n_trials=args.trials, study_name=args.study)
