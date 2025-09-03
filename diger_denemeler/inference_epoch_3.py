#!/usr/bin/env python3
"""
Trendyol E-Ticaret Hackathonu 2025 - Inference Script
two_tower_model_epoch_3.pt kullanarak submission üretir
"""

import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

import polars as pl
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

print("=" * 60)
print("TRENDYOL TWO-TOWER MODEL - INFERENCE (EPOCH 3)")
print("=" * 60)

# GPU kontrolü
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"✅ GPU kullanılıyor: {torch.cuda.get_device_name()}")
    print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("❌ GPU bulunamadı! CPU kullanılacak.")
    device = torch.device('cpu')

print(f"✅ Device: {device}")
print("=" * 60)


class TwoTowerModel(nn.Module):
    """Two-Tower Recommendation Model with Turkish BERT"""
    
    def __init__(self, config):
        super(TwoTowerModel, self).__init__()
        
        self.config = config
        self.bert_model_name = "dbmdz/bert-base-turkish-cased"
        
        print(f"📚 BERT Model yükleniyor: {self.bert_model_name}")
        self.bert = AutoModel.from_pretrained(self.bert_model_name)
        
        # BERT'in sadece son katmanlarını eğit (efficiency için)
        for param in self.bert.parameters():
            param.requires_grad = False
        
        # Sadece son 2 katmanı eğitilebilir yap
        for param in self.bert.encoder.layer[-2:].parameters():
            param.requires_grad = True
            
        self.bert_dim = self.bert.config.hidden_size  # 768
        self.user_embedding_dim = config['user_embedding_dim']
        self.content_embedding_dim = config['content_embedding_dim']
        self.tower_dim = config['tower_dim']
        
        # Embedding layers
        self.user_embedding = nn.Embedding(config['num_users'], self.user_embedding_dim)
        self.content_embedding = nn.Embedding(config['num_contents'], self.content_embedding_dim)
        
        # Query Tower (BERT + User embedding)
        self.query_tower = nn.Sequential(
            nn.Linear(self.bert_dim + self.user_embedding_dim, self.tower_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.tower_dim * 2, self.tower_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Item Tower (Content embedding)
        self.item_tower = nn.Sequential(
            nn.Linear(self.content_embedding_dim, self.tower_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.tower_dim * 2, self.tower_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Output heads
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
        """Initialize weights"""
        for module in [self.user_embedding, self.content_embedding]:
            nn.init.xavier_uniform_(module.weight)
        
        for module in [self.query_tower, self.item_tower, self.click_head, self.order_head]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
    
    def encode_query(self, input_ids, attention_mask, user_ids):
        """Query tower: BERT + User embedding"""
        # BERT encoding with mixed precision
        with torch.cuda.amp.autocast():
            bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            query_text_vec = bert_outputs.last_hidden_state[:, 0, :]  # CLS token
        
        # User embedding
        user_vec = self.user_embedding(user_ids)
        
        # Combine and pass through query tower
        query_input = torch.cat([query_text_vec, user_vec], dim=1)
        query_vec = self.query_tower(query_input)
        
        return query_vec
    
    def encode_item(self, content_ids):
        """Item tower: Content embedding"""
        content_vec = self.content_embedding(content_ids)
        item_vec = self.item_tower(content_vec)
        return item_vec
    
    def forward(self, input_ids, attention_mask, user_ids, content_ids):
        """Forward pass"""
        query_vec = self.encode_query(input_ids, attention_mask, user_ids)
        item_vec = self.encode_item(content_ids)
        
        # Combine vectors
        combined_vec = torch.cat([query_vec, item_vec], dim=1)
        
        # Predictions
        click_logit = self.click_head(combined_vec).squeeze(-1)
        order_logit = self.order_head(combined_vec).squeeze(-1)
        
        return click_logit, order_logit


class TrendyolDataset(Dataset):
    """Dataset for Trendyol recommendation"""
    
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
        
        # Search term tokenization
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
        
        # ID mappings
        user_id = self.user_id_map.get(row['user_id_hashed'], 0)
        content_id = self.content_id_map.get(row['content_id_hashed'], 0)
        
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'user_id': torch.tensor(user_id, dtype=torch.long),
            'content_id': torch.tensor(content_id, dtype=torch.long),
        }
        
        return item


def load_all_data():
    """Load all data sources"""
    print("📂 Veriler yükleniyor...")
    
    base_path = "trendyol-e-ticaret-hackathonu-2025-kaggle/data/"
    
    try:
        # Train ve test sessions
        train_sessions = pl.read_parquet(f"{base_path}train_sessions.parquet")
        test_sessions = pl.read_parquet(f"{base_path}test_sessions.parquet")
        
        print(f"✅ Train sessions: {train_sessions.shape}")
        print(f"✅ Test sessions: {test_sessions.shape}")
        
        # Content data
        content_metadata = pl.read_parquet(f"{base_path}content/metadata.parquet")
        content_price_rate_review = pl.read_parquet(f"{base_path}content/price_rate_review_data.parquet")
        content_search_log = pl.read_parquet(f"{base_path}content/search_log.parquet")
        content_sitewide_log = pl.read_parquet(f"{base_path}content/sitewide_log.parquet")
        content_top_terms = pl.read_parquet(f"{base_path}content/top_terms_log.parquet")
        
        print(f"✅ Content metadata: {content_metadata.shape}")
        print(f"✅ Content price/rate/review: {content_price_rate_review.shape}")
        
        # User data
        user_metadata = pl.read_parquet(f"{base_path}user/metadata.parquet")
        user_search_log = pl.read_parquet(f"{base_path}user/search_log.parquet")
        user_sitewide_log = pl.read_parquet(f"{base_path}user/sitewide_log.parquet")
        user_top_terms = pl.read_parquet(f"{base_path}user/top_terms_log.parquet")
        user_fashion_search = pl.read_parquet(f"{base_path}user/fashion_search_log.parquet")
        user_fashion_sitewide = pl.read_parquet(f"{base_path}user/fashion_sitewide_log.parquet")
        
        print(f"✅ User metadata: {user_metadata.shape}")
        print(f"✅ User search log: {user_search_log.shape}")
        
        # Term data
        term_search_log = pl.read_parquet(f"{base_path}term/search_log.parquet")
        print(f"✅ Term search log: {term_search_log.shape}")
        
        return {
            'train_sessions': train_sessions,
            'test_sessions': test_sessions,
            'content_metadata': content_metadata,
            'content_price_rate_review': content_price_rate_review,
            'content_search_log': content_search_log,
            'content_sitewide_log': content_sitewide_log,
            'content_top_terms': content_top_terms,
            'user_metadata': user_metadata,
            'user_search_log': user_search_log,
            'user_sitewide_log': user_sitewide_log,
            'user_top_terms': user_top_terms,
            'user_fashion_search': user_fashion_search,
            'user_fashion_sitewide': user_fashion_sitewide,
            'term_search_log': term_search_log
        }
    except Exception as e:
        print(f"❌ Veri yükleme hatası: {e}")
        return None


def create_enriched_features(data_dict):
    """Create enriched features from all data sources"""
    print("🔧 Zenginleştirilmiş özellikler oluşturuluyor...")
    
    train_sessions = data_dict['train_sessions']
    test_sessions = data_dict['test_sessions']
    
    # Content features
    content_metadata = data_dict['content_metadata']
    content_price_rate_review = data_dict['content_price_rate_review']
    content_search_log = data_dict['content_search_log']  # ['date', 'total_search_impression', 'total_search_click', 'content_id_hashed']
    content_sitewide_log = data_dict['content_sitewide_log']  # ['date', 'total_click', 'total_cart', 'total_fav', 'total_order', 'content_id_hashed']
    content_top_terms = data_dict['content_top_terms']  # ['date', 'search_term_normalized', 'total_search_impression', 'total_search_click', 'content_id_hashed']
    
    # User features
    user_metadata = data_dict['user_metadata']
    user_search_log = data_dict['user_search_log']  # ['ts_hour', 'total_search_impression', 'total_search_click', 'user_id_hashed']
    user_sitewide_log = data_dict['user_sitewide_log']
    
    # Term features
    term_search_log = data_dict['term_search_log']  # ['date', 'search_term_normalized', 'total_search_impression', 'total_search_click']
    
    # Normalize search terms
    def normalize_search_term(term):
        if term is None:
            return ""
        term_str = str(term).lower().strip()
        if term_str in ['null', 'nan', 'none', '']:
            return ""
        return term_str
    
    # Content features aggregation - use correct column names
    content_features = (
        content_metadata
        .join(content_price_rate_review, on='content_id_hashed', how='left')
        .join(
            content_search_log.group_by('content_id_hashed').agg([
                pl.sum('total_search_impression').alias('search_frequency'),
                pl.sum('total_search_click').alias('search_clicks')
            ]), on='content_id_hashed', how='left'
        )
        .join(
            content_sitewide_log.group_by('content_id_hashed').agg([
                pl.sum('total_click').alias('sitewide_clicks'),
                pl.sum('total_cart').alias('sitewide_carts'),
                pl.sum('total_fav').alias('sitewide_favs'),
                pl.sum('total_order').alias('sitewide_orders')
            ]), on='content_id_hashed', how='left'
        )
        .join(
            content_top_terms.group_by('content_id_hashed').agg([
                pl.n_unique('search_term_normalized').alias('unique_search_terms')
            ]), on='content_id_hashed', how='left'
        )
        .with_columns([
            pl.col('search_term_normalized').map_elements(normalize_search_term, return_dtype=pl.String).alias('search_term_normalized')
        ])
    )
    
    # User features aggregation - use correct column names
    user_features = (
        user_metadata
        .join(
            user_search_log.group_by('user_id_hashed').agg([
                pl.sum('total_search_impression').alias('user_search_impressions'),
                pl.sum('total_search_click').alias('user_search_clicks')
            ]), on='user_id_hashed', how='left'
        )
        .join(
            user_sitewide_log.group_by('user_id_hashed').agg([
                pl.sum('total_click').alias('user_sitewide_clicks'),
                pl.sum('total_cart').alias('user_sitewide_carts'),
                pl.sum('total_fav').alias('user_sitewide_favs'),
                pl.sum('total_order').alias('user_sitewide_orders')
            ]), on='user_id_hashed', how='left'
        )
    )
    
    # Fashion features (if available)
    try:
        user_fashion_features = (
            data_dict['user_fashion_search']
            .group_by('user_id_hashed').agg([
                pl.sum('total_search_impression').alias('user_fashion_search_impressions'),
                pl.sum('total_search_click').alias('user_fashion_search_clicks'),
                pl.n_unique('content_id_hashed').alias('user_fashion_unique_contents')
            ])
        )
        user_features = user_features.join(user_fashion_features, on='user_id_hashed', how='left')
        
        user_fashion_sitewide_features = (
            data_dict['user_fashion_sitewide']
            .group_by('user_id_hashed').agg([
                pl.sum('total_click').alias('user_fashion_sitewide_clicks'),
                pl.sum('total_cart').alias('user_fashion_sitewide_carts'),
                pl.sum('total_fav').alias('user_fashion_sitewide_favs'),
                pl.sum('total_order').alias('user_fashion_sitewide_orders')
            ])
        )
        user_features = user_features.join(user_fashion_sitewide_features, on='user_id_hashed', how='left')
        
        print("✅ Fashion features added")
    except Exception as e:
        print(f"⚠️ Fashion verileri eklenirken hata: {e}")
    
    # Term features - use correct column names
    term_features = (
        term_search_log
        .with_columns([
            pl.col('search_term_normalized').map_elements(normalize_search_term, return_dtype=pl.String).alias('search_term_normalized')
        ])
        .group_by('search_term_normalized').agg([
            pl.sum('total_search_impression').alias('term_frequency'),
            pl.sum('total_search_click').alias('term_clicks')
        ])
    )
    
    # Enrich train data
    train_enriched = (
        train_sessions
        .with_columns([
            pl.col('search_term_normalized').map_elements(normalize_search_term, return_dtype=pl.String).alias('search_term_normalized')
        ])
        .join(content_features, on='content_id_hashed', how='left')
        .join(user_features, on='user_id_hashed', how='left') 
        .join(term_features, on='search_term_normalized', how='left')
    )
    
    # Enrich test data
    test_enriched = (
        test_sessions
        .with_columns([
            pl.col('search_term_normalized').map_elements(normalize_search_term, return_dtype=pl.String).alias('search_term_normalized')
        ])
        .join(content_features, on='content_id_hashed', how='left')
        .join(user_features, on='user_id_hashed', how='left')
        .join(term_features, on='search_term_normalized', how='left')
    )
    
    print(f"✅ Zenginleştirilmiş train data: {train_enriched.shape}")
    print(f"✅ Zenginleştirilmiş test data: {test_enriched.shape}")
    
    return train_enriched, test_enriched


def create_id_mappings(train_df, test_df):
    """Create ID mappings"""
    print("🗺️ ID mappings oluşturuluyor...")
    
    # User ID mapping
    all_users = pd.concat([train_df['user_id_hashed'], test_df['user_id_hashed']]).unique()
    user_id_map = {user_id: idx + 1 for idx, user_id in enumerate(all_users)}
    
    # Content ID mapping
    all_contents = pd.concat([train_df['content_id_hashed'], test_df['content_id_hashed']]).unique()
    content_id_map = {content_id: idx + 1 for idx, content_id in enumerate(all_contents)}
    
    print(f"✅ User mapping: {len(user_id_map):,} users")
    print(f"✅ Content mapping: {len(content_id_map):,} contents")
    
    return user_id_map, content_id_map


def load_checkpoint(checkpoint_path, device):
    """Load model checkpoint"""
    print(f"📥 Checkpoint yükleniyor: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint bulunamadı: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    print(f"✅ Epoch: {checkpoint['epoch']}")
    print(f"✅ Validation Combined Score: {checkpoint['val_combined_score']:.4f}")
    print(f"✅ Click AUC: {checkpoint['val_click_auc']:.4f}")
    print(f"✅ Order AUC: {checkpoint['val_order_auc']:.4f}")
    
    return checkpoint


def main():
    """Main inference function"""
    print("🚀 Two-Tower Model Inference - Epoch 3...")
    
    # Checkpoint path
    checkpoint_path = 'two_tower_model_epoch_3.pt'
    
    # Load checkpoint
    try:
        checkpoint = load_checkpoint(checkpoint_path, device)
    except FileNotFoundError:
        print(f"❌ Checkpoint bulunamadı: {checkpoint_path}")
        print("📋 Mevcut .pt dosyaları:")
        pt_files = [f for f in os.listdir('.') if f.endswith('.pt')]
        for pt_file in pt_files:
            print(f"  - {pt_file}")
        return
    
    # Load data
    data_dict = load_all_data()
    if data_dict is None:
        print("❌ Veri yüklenemedi!")
        return
    
    # Create enriched features
    train_enriched, test_enriched = create_enriched_features(data_dict)
    
    # Convert to pandas
    print("📊 Veri pandas'a çevriliyor...")
    train_df = train_enriched.to_pandas()
    test_df = test_enriched.to_pandas()
    
    # Fill missing values
    print("🔧 Eksik değerler dolduruluyor...")
    
    # Numeric columns
    train_numeric_columns = train_df.select_dtypes(include=[np.number]).columns
    for col in train_numeric_columns:
        train_df[col] = train_df[col].fillna(0)
    
    test_numeric_columns = test_df.select_dtypes(include=[np.number]).columns
    for col in test_numeric_columns:
        test_df[col] = test_df[col].fillna(0)
    
    # Categorical columns
    train_categorical_columns = train_df.select_dtypes(include=['object']).columns
    for col in train_categorical_columns:
        if col not in ['user_id_hashed', 'content_id_hashed', 'session_id']:
            train_df[col] = train_df[col].fillna('unknown')
    
    test_categorical_columns = test_df.select_dtypes(include=['object']).columns
    for col in test_categorical_columns:
        if col not in ['user_id_hashed', 'content_id_hashed', 'session_id']:
            test_df[col] = test_df[col].fillna('unknown')
    
    # Search term normalization
    train_df['search_term_normalized'] = train_df['search_term_normalized'].fillna('')
    test_df['search_term_normalized'] = test_df['search_term_normalized'].fillna('')
    
    print(f"✅ Train data shape: {train_df.shape}")
    print(f"✅ Test data shape: {test_df.shape}")
    
    # Create ID mappings
    user_id_map = checkpoint['user_id_map']
    content_id_map = checkpoint['content_id_map']
    
    # Model configuration
    model_config = checkpoint['model_config']
    print(f"📋 Model config: {model_config}")
    
    # Initialize model
    print("🏗️ Model oluşturuluyor...")
    model = TwoTowerModel(model_config).to(device)
    tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model yüklendi (Epoch {checkpoint['epoch']})")
    print(f"✅ Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create test dataset
    print("📝 Test dataset oluşturuluyor...")
    test_dataset = TrendyolDataset(test_df, tokenizer, user_id_map, content_id_map, max_length=128)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=96,
        shuffle=False, 
        num_workers=0,
        pin_memory=True
    )
    
    # Inference
    print(f"🔮 Test inference başlıyor: {len(test_loader)} batches")
    
    all_click_preds = []
    all_order_preds = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx % 200 == 0:
                print(f"Batch {batch_idx+1}/{len(test_loader)} | "
                      f"GPU Memory: {torch.cuda.memory_allocated()/1024**3:.1f}GB")
            
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
    
    # Create predictions
    click_pred_test = np.array(all_click_preds)
    order_pred_test = np.array(all_order_preds)
    combined_pred_test = 0.3 * click_pred_test + 0.7 * order_pred_test
    
    print(f"✅ Click predictions - min: {click_pred_test.min():.4f}, max: {click_pred_test.max():.4f}, mean: {click_pred_test.mean():.4f}")
    print(f"✅ Order predictions - min: {order_pred_test.min():.4f}, max: {order_pred_test.max():.4f}, mean: {order_pred_test.mean():.4f}")
    print(f"✅ Combined predictions - min: {combined_pred_test.min():.4f}, max: {combined_pred_test.max():.4f}, mean: {combined_pred_test.mean():.4f}")
    
    # Create submission
    test_predictions = pd.DataFrame({
        'session_id': test_df['session_id'].values,
        'content_id_hashed': test_df['content_id_hashed'].values,
        'combined_score': combined_pred_test
    })
    
    # Create final submission format
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
    
    # Save submission with epoch info
    submission_filename = f'submission_epoch_3.csv'
    final_submission.to_csv(submission_filename, index=False)
    
    print(f"✅ Submission saved: {submission_filename}")
    print(f"✅ Sessions: {len(final_submission):,}")
    print(f"✅ Total predictions: {len(test_predictions):,}")
    
    # Validation check
    try:
        sample_submission = pd.read_csv('trendyol-e-ticaret-hackathonu-2025-kaggle/data/sample_submission.csv')
        print(f"✅ Sample sessions: {len(sample_submission):,}")
        print(f"✅ Our sessions: {len(final_submission):,}")
        
        if len(final_submission) == len(sample_submission):
            print("✅ Session count PERFECT MATCH!")
        else:
            print("⚠️ Session count mismatch!")
    except Exception as e:
        print(f"⚠️ Could not validate submission: {e}")
    
    print("\n🎯 INFERENCE WITH EPOCH 3 MODEL - SUCCESS!")
    print(f"📚 Model Epoch: {checkpoint['epoch']}")
    print(f"🔥 Validation Score: {checkpoint['val_combined_score']:.4f}")
    print(f"💎 Submission: {submission_filename}")
    print("=" * 60)


if __name__ == "__main__":
    main()
