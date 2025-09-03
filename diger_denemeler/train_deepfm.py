"""
Script to train a DeepFM model on preprocessed session data.

Usage:
  python tools/train_deepfm.py

Requires:
  torch, numpy, pandas
"""
import json
import pickle
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torch.cuda.amp import GradScaler, autocast
import warnings
from sklearn.exceptions import UndefinedMetricWarning
# Suppress undefined metric warnings from sklearn
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


class SessionDeepFMDataset(Dataset):
    def __init__(self, X_cat, X_num, y1, y2):
        self.X_cat = X_cat.astype(np.int64)
        self.X_num = X_num.astype(np.float32)
        self.y1 = y1.astype(np.float32)
        self.y2 = y2.astype(np.float32)

    def __len__(self):
        return len(self.y1)

    def __getitem__(self, idx):
        return {
            'cat': torch.from_numpy(self.X_cat[idx]),
            'num': torch.from_numpy(self.X_num[idx]),
            'y1': torch.tensor(self.y1[idx]),
            'y2': torch.tensor(self.y2[idx]),
        }


class DeepFM(nn.Module):
    def __init__(self, feature_sizes, num_num, embed_dim=16, hidden_dims=[256,128,64], dropout=0.3):
        super().__init__()
        self.num_cat = len(feature_sizes)
        self.num_num = num_num
        self.embed_dim = embed_dim
        # first-order embeddings for categorical
        self.emb0 = nn.ModuleList([nn.Embedding(size, 1) for size in feature_sizes])
        # first-order weights for numeric
        self.num_bias = nn.Parameter(torch.zeros(num_num, 1))
        # second-order embeddings for categorical
        self.emb1 = nn.ModuleList([nn.Embedding(size, embed_dim) for size in feature_sizes])
        # embeddings for numeric numeric features for FM and deep
        self.num_emb = nn.Parameter(torch.randn(num_num, embed_dim) * 0.01)
        # deep part
        input_dim = (self.num_cat + self.num_num) * embed_dim
        layers = []
        for dim in hidden_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = dim
        self.mlp = nn.Sequential(*layers)
        # task heads
        self.fc_click = nn.Linear(input_dim + 2, 1)  # +2 for linear + FM outputs
        self.fc_order = nn.Linear(input_dim + 2, 1)

    def forward(self, x_cat, x_num):
        # linear terms
        linear_cat = [emb0(x_cat[:, i].unsqueeze(1)).squeeze(1) for i, emb0 in enumerate(self.emb0)]
        linear_cat = torch.cat(linear_cat, dim=1)  # (batch, num_cat)
        linear_num = x_num.unsqueeze(2) * self.num_bias.unsqueeze(0)  # (batch,num_num,1)
        linear_num = linear_num.squeeze(2)  # (batch,num_num)
        linear_term = torch.sum(torch.cat([linear_cat, linear_num], dim=1), dim=1, keepdim=True)  # (batch,1)
        # FM second-order
        # embeddings
        cat_emb = [emb1(x_cat[:, i].unsqueeze(1)).squeeze(1) for i, emb1 in enumerate(self.emb1)]
        cat_emb = torch.stack(cat_emb, dim=1)  # (batch, num_cat, embed_dim)
        num_emb = x_num.unsqueeze(2) * self.num_emb.unsqueeze(0)  # (batch, num_num, embed_dim)
        all_emb = torch.cat([cat_emb, num_emb], dim=1)  # (batch, fields, embed_dim)
        sum_emb = torch.sum(all_emb, dim=1)  # (batch,embed_dim)
        sum_emb_square = sum_emb * sum_emb  # (batch,embed_dim)
        square_emb = all_emb * all_emb
        square_sum_emb = torch.sum(square_emb, dim=1)  # (batch,embed_dim)
        fm_second = 0.5 * torch.sum(sum_emb_square - square_sum_emb, dim=1, keepdim=True)  # (batch,1)
        # deep part
        deep_input = all_emb.view(all_emb.size(0), -1)  # (batch, fields*embed_dim)
        deep_out = self.mlp(deep_input)  # (batch, last_dim)
        # concatenate linear + fm
        concat = torch.cat([deep_out, linear_term, fm_second], dim=1)
        # logits
        logit1 = self.fc_click(concat)
        logit2 = self.fc_order(concat)
        return logit1, logit2


def train():
    # Argument parser for hyperparams
    parser = argparse.ArgumentParser()
    parser.add_argument('--patience', type=int, default=3, help='EarlyStopping patience')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--prefetch', type=int, default=2)
    args = parser.parse_args()
    # Paths
    ROOT = Path(__file__).resolve().parents[1]
    DP = ROOT / "trendyol-e-ticaret-hackathonu-2025-kaggle" / "data" / "deepfm"
    # load meta
    meta = json.loads((DP / 'meta_deepfm.json').read_text())
    cat_cols = meta['cat_cols']
    num_cols = meta['num_cols']
    full_feature_sizes = meta['feature_sizes']
    # Meta stores sizes for categorical embeddings followed by numeric placeholders
    n_cat = len(meta['cat_cols'])
    feature_sizes = full_feature_sizes[:n_cat]
    # load arrays
    X_train_all = np.load(DP / 'X_train.npy')
    y_click = np.load(DP / 'y_train.npy')
    y_order = np.load(DP / 'y_order.npy')
    # define device early for weighting and model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    # split features into categorical and numeric
    n_cat = len(cat_cols)
    X_train_cat = X_train_all[:, :n_cat]
    X_train_num = X_train_all[:, n_cat:]
    # compute data size
    n_train = len(y_click)
    # compute class weights for balance
    neg_click = np.sum(y_click == 0)
    pos_click = np.sum(y_click == 1)
    pos_weight_click = torch.tensor(neg_click / (pos_click + 1), dtype=torch.float32, device=device)
    neg_order = np.sum(y_order == 0)
    pos_order = np.sum(y_order == 1)
    pos_weight_order = torch.tensor(neg_order / (pos_order + 1), dtype=torch.float32, device=device)
    # define loss functions with pos_weight
    criterion_click = nn.BCEWithLogitsLoss(pos_weight=pos_weight_click)
    criterion_order = nn.BCEWithLogitsLoss(pos_weight=pos_weight_order)
    # train/val split
    from sklearn.model_selection import train_test_split
    X_cat_train, X_cat_val, X_num_train, X_num_val, y1_train, y1_val, y2_train, y2_val = \
        train_test_split(X_train_cat, X_train_num, y_click, y_order, test_size=0.1, random_state=42)
    train_set = SessionDeepFMDataset(X_cat_train, X_num_train, y1_train, y2_train)
    val_set = SessionDeepFMDataset(X_cat_val, X_num_val, y1_val, y2_val)
    # model and device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    model = DeepFM(feature_sizes, len(num_cols)).to(device)
    # DataLoaders with optimizations
    dl_train = DataLoader(train_set, batch_size=512, shuffle=True,
                          pin_memory=(device.type=='cuda'), num_workers=args.num_workers,
                          persistent_workers=True, prefetch_factor=args.prefetch)
    dl_val = DataLoader(val_set, batch_size=512,
                        pin_memory=(device.type=='cuda'), num_workers=args.num_workers,
                        persistent_workers=True, prefetch_factor=args.prefetch)
    # prepare scaler for mixed precision
    scaler = GradScaler()
    # training loop with early stopping & mixed precision
    best_val_score = -float('inf')
    epochs_no_improve = 0
    for epoch in range(1, 11):
        model.train()
        total_loss = 0.0
        # training batches
        for batch in dl_train:
            x_cat = batch['cat'].to(device)
            x_num = batch['num'].to(device)
            y1 = batch['y1'].to(device).unsqueeze(1)
            y2 = batch['y2'].to(device).unsqueeze(1)
            optimizer.zero_grad()
            with autocast():
                logit1, logit2 = model(x_cat, x_num)
                loss1 = criterion_click(logit1, y1)
                loss2 = criterion_order(logit2, y2)
                loss = loss1 + loss2
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * x_cat.size(0)
        avg_train_loss = total_loss / n_train
        # validation with same autocast
        model.eval()
        all_y1, all_p1, all_y2, all_p2 = [], [], [], []
        with torch.no_grad():
            for batch in dl_val:
                x_cat = batch['cat'].to(device)
                x_num = batch['num'].to(device)
                with autocast():
                    logit1, logit2 = model(x_cat, x_num)
                p1 = torch.sigmoid(logit1).cpu().numpy().flatten()
                p2 = torch.sigmoid(logit2).cpu().numpy().flatten()
                all_p1.extend(p1); all_p2.extend(p2)
                all_y1.extend(batch['y1'].numpy()); all_y2.extend(batch['y2'].numpy())
        auc1 = metrics.roc_auc_score(all_y1, all_p1)
        auc2 = metrics.roc_auc_score(all_y2, all_p2)
        val_score = auc1 + auc2
        # early stopping check
        if val_score > best_val_score:
            best_val_score = val_score
            epochs_no_improve = 0
            torch.save(model.state_dict(), DP / 'best_deepfm.pt')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"No improvement for {args.patience} epochs. Early stopping at epoch {epoch}.")
                break
        # log results
        print(
            f"Epoch {epoch}: Train Loss={avg_train_loss:.4f} | "
            f"Val AUC_click={auc1:.4f}, Prec={pr1:.4f}, Rec={rc1:.4f}, F1={f1_1:.4f} | "
            f"AUC_order={auc2:.4f}, Prec={pr2:.4f}, Rec={rc2:.4f}, F1={f1_2:.4f}"
        )
    # end training loop
    print(f"Training complete. Best Val Score={best_val_score:.4f}")
    # Detailed evaluation on validation set
    model.eval()
    import warnings
    import sklearn.metrics as metrics
    # suppress precision undefined warnings
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    all_y1, all_p1, all_y2, all_p2 = [], [], [], []
    with torch.no_grad():
        for batch in dl_val:
            x_cat = batch['cat'].to(device)
            x_num = batch['num'].to(device)
            y1 = batch['y1'].cpu().numpy()
            y2 = batch['y2'].cpu().numpy()
            logit1, logit2 = model(x_cat, x_num)
            p1 = torch.sigmoid(logit1).cpu().numpy().flatten()
            p2 = torch.sigmoid(logit2).cpu().numpy().flatten()
            all_y1.extend(y1)
            all_p1.extend(p1)
            all_y2.extend(y2)
            all_p2.extend(p2)
    pred1 = np.array(all_p1) > 0.5
    pred2 = np.array(all_p2) > 0.5
    # Prepare int arrays for classification report
    y1_arr = np.array(all_y1, dtype=int)
    pred1_arr = pred1.astype(int)
    y2_arr = np.array(all_y2, dtype=int)
    pred2_arr = pred2.astype(int)
    print("\nValidation Classification Report for Click Task:")
    # Provide named classes for display
    print(metrics.classification_report(y1_arr, pred1_arr,
                                        target_names=['no_click', 'click'], zero_division=0))
    print("Validation Classification Report for Order Task:")
    # Provide named classes for display
    print(metrics.classification_report(y2_arr, pred2_arr,
                                        target_names=['no_order', 'order'], zero_division=0))


def main():
    train()


if __name__ == '__main__':
    main()
