#!/usr/bin/env python3
"""
Trendyol E-Ticaret Hackathonu 2025 - Simple Inference Script
two_tower_model_epoch_3.pt kullanarak submission üretir (Minimal version)
"""

import os
import warnings
warnings.filterwarnings('ignore')

import polars as pl
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

print("=" * 60)
print("TRENDYOL TWO-TOWER MODEL - SIMPLE INFERENCE (EPOCH 3)")
print("=" * 60)

# GPU kontrolü
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"✅ GPU kullanılıyor: {torch.cuda.get_device_name()}")
else:
    device = torch.device('cpu')
    print("❌ GPU bulunamadı! CPU kullanılacak.")

print(f"✅ Device: {device}")


class TwoTowerModel(nn.Module):
    """Two-Tower Recommendation Model with Turkish BERT"""
    
    def __init__(self, config):
        super(TwoTowerModel, self).__init__()
        
        self.config = config
        self.bert_model_name = "dbmdz/bert-base-turkish-cased"
        
        print(f"📚 BERT Model yükleniyor: {self.bert_model_name}")
        self.bert = AutoModel.from_pretrained(self.bert_model_name)
        
        for param in self.bert.parameters():
            param.requires_grad = False
        for param in self.bert.encoder.layer[-2:].parameters():
            param.requires_grad = True
            
        self.bert_dim = self.bert.config.hidden_size
        self.user_embedding_dim = config['user_embedding_dim']
        self.content_embedding_dim = config['content_embedding_dim']
        self.tower_dim = config['tower_dim']
        
        self.user_embedding = nn.Embedding(config['num_users'], self.user_embedding_dim)
        self.content_embedding = nn.Embedding(config['num_contents'], self.content_embedding_dim)
        
        self.query_tower = nn.Sequential(
            nn.Linear(self.bert_dim + self.user_embedding_dim, self.tower_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.tower_dim * 2, self.tower_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.item_tower = nn.Sequential(
            nn.Linear(self.content_embedding_dim, self.tower_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.tower_dim * 2, self.tower_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.click_head = nn.Sequential(
            nn.Linear(self.tower_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
        
        self.order_head = nn.Sequential(
            nn.Linear(self.tower_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in [self.user_embedding, self.content_embedding]:
            nn.init.xavier_uniform_(module.weight)
        
        for module in [self.query_tower, self.item_tower, self.click_head, self.order_head]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
    
    def encode_query(self, input_ids, attention_mask, user_ids):
        with torch.cuda.amp.autocast():
            bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            query_text_vec = bert_outputs.last_hidden_state[:, 0, :]
        
        user_vec = self.user_embedding(user_ids)
        query_input = torch.cat([query_text_vec, user_vec], dim=1)
        query_vec = self.query_tower(query_input)
        return query_vec
    
    def encode_item(self, content_ids):
        content_vec = self.content_embedding(content_ids)
        item_vec = self.item_tower(content_vec)
        return item_vec
    
    def forward(self, input_ids, attention_mask, user_ids, content_ids):
        query_vec = self.encode_query(input_ids, attention_mask, user_ids)
        item_vec = self.encode_item(content_ids)
        combined_vec = torch.cat([query_vec, item_vec], dim=1)
        click_logit = self.click_head(combined_vec).squeeze(-1)
        order_logit = self.order_head(combined_vec).squeeze(-1)
        return click_logit, order_logit


class TrendyolDataset(Dataset):
    def __init__(self, data_df, tokenizer, user_id_map, content_id_map, max_length=128):
        self.data = data_df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.user_id_map = user_id_map
        self.content_id_map = content_id_map
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        search_term = str(row.get('search_term_normalized', ''))
        if pd.isna(search_term) or search_term in ['nan', 'null', 'None']:
            search_term = ''
            
        encoding = self.tokenizer(
            search_term,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        user_id = self.user_id_map.get(row['user_id_hashed'], 0)
        content_id = self.content_id_map.get(row['content_id_hashed'], 0)
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'user_id': torch.tensor(user_id, dtype=torch.long),
            'content_id': torch.tensor(content_id, dtype=torch.long),
        }


def main():
    """Simple inference function"""
    print("🚀 Simple Two-Tower Model Inference - Epoch 3...")
    
    # Checkpoint yükleme
    checkpoint_path = 'two_tower_model_epoch_3.pt'
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint bulunamadı: {checkpoint_path}")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(f"✅ Epoch: {checkpoint['epoch']}")
    print(f"✅ Validation Combined Score: {checkpoint['val_combined_score']:.4f}")
    
    # Sadece test_sessions yükle
    print("📂 Test sessions yükleniyor...")
    base_path = "trendyol-e-ticaret-hackathonu-2025-kaggle/data/"
    test_sessions = pl.read_parquet(f"{base_path}test_sessions.parquet").to_pandas()
    
    # Eksik değerleri doldur
    test_sessions['search_term_normalized'] = test_sessions['search_term_normalized'].fillna('')
    
    print(f"✅ Test data shape: {test_sessions.shape}")
    
    # Checkpoint'tan ID mappings
    user_id_map = checkpoint['user_id_map']
    content_id_map = checkpoint['content_id_map']
    model_config = checkpoint['model_config']
    
    print(f"📋 Model config: {model_config}")
    
    # Model yükleme
    print("🏗️ Model oluşturuluyor...")
    model = TwoTowerModel(model_config).to(device)
    tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model yüklendi (Epoch {checkpoint['epoch']})")
    
    # Dataset ve DataLoader
    print("📝 Test dataset oluşturuluyor...")
    test_dataset = TrendyolDataset(test_sessions, tokenizer, user_id_map, content_id_map, max_length=128)
    test_loader = DataLoader(test_dataset, batch_size=96, shuffle=False, num_workers=0, pin_memory=True)
    
    # Inference
    print(f"🔮 Test inference başlıyor: {len(test_loader)} batches")
    
    all_click_preds = []
    all_order_preds = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx % 200 == 0:
                print(f"Batch {batch_idx+1}/{len(test_loader)}")
            
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            user_ids = batch['user_id'].to(device, non_blocking=True)
            content_ids = batch['content_id'].to(device, non_blocking=True)
            
            with torch.cuda.amp.autocast():
                click_logits, order_logits = model(input_ids, attention_mask, user_ids, content_ids)
            
            click_probs = torch.sigmoid(click_logits).cpu().numpy()
            order_probs = torch.sigmoid(order_logits).cpu().numpy()
            
            all_click_preds.extend(click_probs)
            all_order_preds.extend(order_probs)
    
    # Predictions oluştur
    click_pred_test = np.array(all_click_preds)
    order_pred_test = np.array(all_order_preds)
    combined_pred_test = 0.3 * click_pred_test + 0.7 * order_pred_test
    
    print(f"✅ Combined predictions - min: {combined_pred_test.min():.4f}, max: {combined_pred_test.max():.4f}, mean: {combined_pred_test.mean():.4f}")
    
    # Submission oluştur
    test_predictions = pd.DataFrame({
        'session_id': test_sessions['session_id'].values,
        'content_id_hashed': test_sessions['content_id_hashed'].values,
        'combined_score': combined_pred_test
    })
    
    print("📝 Final submission oluşturuluyor...")
    submission_list = []
    
    unique_sessions = test_predictions['session_id'].unique()
    print(f"✅ Unique sessions: {len(unique_sessions):,}")
    
    for i, session_id in enumerate(unique_sessions):
        if i % 10000 == 0:
            print(f"İşlenen session: {i+1:,}/{len(unique_sessions):,}")
            
        session_data = test_predictions[test_predictions['session_id'] == session_id].sort_values(
            'combined_score', ascending=False
        )
        content_ids_string = ' '.join(session_data['content_id_hashed'].astype(str))
        submission_list.append({
            'session_id': session_id,
            'prediction': content_ids_string
        })
    
    final_submission = pd.DataFrame(submission_list)
    
    # Submission kaydet
    submission_filename = 'submission_final_correct.csv'
    final_submission.to_csv(submission_filename, index=False)
    
    print(f"✅ Submission saved: {submission_filename}")
    print(f"✅ Sessions: {len(final_submission):,}")
    print(f"✅ Total predictions: {len(test_predictions):,}")
    
    # Validation check - sample_submission ile karşılaştır
    try:
        sample_submission = pd.read_csv('trendyol-e-ticaret-hackathonu-2025-kaggle/data/sample_submission.csv')
        print(f"✅ Sample sessions: {len(sample_submission):,}")
        print(f"✅ Our sessions: {len(final_submission):,}")
        
        if len(final_submission) == len(sample_submission):
            print("🎯 Session count PERFECT MATCH!")
        else:
            print("⚠️ Session count mismatch!")
            
        # Column kontrolü
        if set(final_submission.columns) == set(sample_submission.columns):
            print("✅ Column names PERFECT MATCH!")
        else:
            print(f"⚠️ Columns mismatch - Sample: {list(sample_submission.columns)}, Ours: {list(final_submission.columns)}")
            
    except Exception as e:
        print(f"⚠️ Could not validate submission: {e}")
    
    print("\n🎯 SIMPLE INFERENCE WITH EPOCH 3 MODEL - SUCCESS!")
    print(f"📚 Model Epoch: {checkpoint['epoch']}")
    print(f"🔥 Validation Score: {checkpoint['val_combined_score']:.4f}")
    print(f"💎 Submission: {submission_filename}")
    print("=" * 60)


if __name__ == "__main__":
    main()
