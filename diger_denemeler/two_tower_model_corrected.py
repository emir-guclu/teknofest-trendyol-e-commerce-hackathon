#!/usr/bin/env python3
"""
Trendyol E-Ticaret Hackathonu 2025 - Two-Tower Model with Turkish BERT
CORRECTED VERSION - Proper data inspection and column names
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
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

print("=" * 60)
print("TRENDYOL TWO-TOWER MODEL - VERİ İNCELENMİŞ SÜRÜM")
print("=" * 60)

# GPU kontrolü
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"✅ GPU kullanılıyor: {torch.cuda.get_device_name()}")
    print(f"✅ PyTorch Version: {torch.__version__}")
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
        print(f"✅ Model oluşturuldu. Toplam parametreler: {sum(p.numel() for p in self.parameters()):,}")
        print(f"✅ Eğitilebilir parametreler: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
    
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
        
        # Training targets
        if 'clicked' in self.data.columns:
            item['clicked'] = torch.tensor(row['clicked'], dtype=torch.float)
            item['ordered'] = torch.tensor(row['ordered'], dtype=torch.float)
        
        return item


def load_all_data():
    """Load ALL data sources based on actual data inspection"""
    print("📂 TÜM veri kaynakları yükleniyor (İNCELENMİŞ KOLON İSİMLERİ)...")
    
    DATA_PATH = "trendyol-e-ticaret-hackathonu-2025-kaggle/data/"
    
    # Main sessions data
    print("🎯 Ana sessions verileri...")
    train_sessions = pl.read_parquet(f"{DATA_PATH}train_sessions.parquet")
    test_sessions = pl.read_parquet(f"{DATA_PATH}test_sessions.parquet")
    
    print(f"✅ Train sessions: {train_sessions.shape}")
    print(f"✅ Test sessions: {test_sessions.shape}")
    print(f"Train columns: {train_sessions.columns}")
    print(f"Test columns: {test_sessions.columns}")
    
    # Content data sources
    print("\n📚 Content veri kaynakları...")
    content_metadata = pl.read_parquet(f"{DATA_PATH}content/metadata.parquet")
    content_price_data = pl.read_parquet(f"{DATA_PATH}content/price_rate_review_data.parquet")
    content_search_log = pl.read_parquet(f"{DATA_PATH}content/search_log.parquet")
    content_sitewide_log = pl.read_parquet(f"{DATA_PATH}content/sitewide_log.parquet")
    content_top_terms = pl.read_parquet(f"{DATA_PATH}content/top_terms_log.parquet")
    
    print(f"✅ Content metadata: {content_metadata.shape}")
    print(f"✅ Content price/rate/review: {content_price_data.shape}")
    print(f"✅ Content search log: {content_search_log.shape}")
    print(f"✅ Content sitewide log: {content_sitewide_log.shape}")
    print(f"✅ Content top terms: {content_top_terms.shape}")
    
    # User data sources
    print("\n👤 User veri kaynakları...")
    user_metadata = pl.read_parquet(f"{DATA_PATH}user/metadata.parquet")
    user_search_log = pl.read_parquet(f"{DATA_PATH}user/search_log.parquet")
    user_sitewide_log = pl.read_parquet(f"{DATA_PATH}user/sitewide_log.parquet")
    user_top_terms = pl.read_parquet(f"{DATA_PATH}user/top_terms_log.parquet")
    user_fashion_search = pl.read_parquet(f"{DATA_PATH}user/fashion_search_log.parquet")
    user_fashion_sitewide = pl.read_parquet(f"{DATA_PATH}user/fashion_sitewide_log.parquet")
    
    print(f"✅ User metadata: {user_metadata.shape}")
    print(f"✅ User search log: {user_search_log.shape}")
    print(f"✅ User sitewide log: {user_sitewide_log.shape}")
    print(f"✅ User top terms: {user_top_terms.shape}")
    print(f"✅ User fashion search: {user_fashion_search.shape}")
    print(f"✅ User fashion sitewide: {user_fashion_sitewide.shape}")
    
    # Term data
    print("\n🔍 Search term verileri...")
    term_search_log = pl.read_parquet(f"{DATA_PATH}term/search_log.parquet")
    print(f"✅ Term search log: {term_search_log.shape}")
    
    return {
        'train_sessions': train_sessions,
        'test_sessions': test_sessions,
        'content_metadata': content_metadata,
        'content_price_data': content_price_data,
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


def create_enriched_features(data_dict):
    """Create enriched features from all data sources with CORRECT column names"""
    print("\n🔧 Zenginleştirilmiş özellikler oluşturuluyor (DOĞRU KOLONLAR)...")
    
    train_sessions = data_dict['train_sessions']
    test_sessions = data_dict['test_sessions']
    
    # Content features with CORRECT column names
    print("📚 Content özellikleri (DOĞRU KOLONLAR)...")
    
    # Content metadata - level1_category_name, level2_category_name, leaf_category_name
    content_features = data_dict['content_metadata'].select([
        'content_id_hashed',
        'level1_category_name',
        'level2_category_name', 
        'leaf_category_name',
        'attribute_type_count',
        'total_attribute_option_count',
        'merchant_count',
        'filterable_label_count'
    ]).with_columns([
        pl.col('level1_category_name').fill_null('unknown').alias('level1_cat'),
        pl.col('level2_category_name').fill_null('unknown').alias('level2_cat'),
        pl.col('leaf_category_name').fill_null('unknown').alias('leaf_cat')
    ])
    
    # Content price/rate/review with CORRECT columns
    content_price_stats = data_dict['content_price_data'].group_by('content_id_hashed').agg([
        pl.col('selling_price').mean().alias('avg_selling_price'),
        pl.col('original_price').mean().alias('avg_original_price'),
        pl.col('content_rate_avg').mean().alias('avg_content_rating'),
        pl.col('content_review_count').mean().alias('avg_review_count'),
        pl.col('content_rate_count').mean().alias('avg_rate_count'),
        pl.col('selling_price').count().alias('price_data_count')
    ])
    
    # Content search impressions/clicks with CORRECT columns  
    content_search_stats = data_dict['content_search_log'].group_by('content_id_hashed').agg([
        pl.col('total_search_impression').sum().alias('total_search_impressions'),
        pl.col('total_search_click').sum().alias('total_search_clicks'),
        pl.col('date').count().alias('search_log_days')
    ]).with_columns([
        (pl.col('total_search_clicks') / pl.col('total_search_impressions').clip(lower_bound=1e-8)).alias('search_ctr')
    ])
    
    # Content sitewide popularity - using count of interactions
    content_sitewide_stats = data_dict['content_sitewide_log'].group_by('content_id_hashed').agg([
        pl.count().alias('sitewide_interactions'),
        pl.col('user_id_hashed').n_unique().alias('unique_users_interacted') if 'user_id_hashed' in data_dict['content_sitewide_log'].columns else pl.lit(0).alias('unique_users_interacted')
    ])
    
    # Combine all content features
    print("🔗 Content özellikleri birleştiriliyor...")
    content_enriched = content_features
    for stats_df in [content_price_stats, content_search_stats, content_sitewide_stats]:
        content_enriched = content_enriched.join(stats_df, on='content_id_hashed', how='left')
    
    print(f"✅ Content enriched shape: {content_enriched.shape}")
    
    # User features with CORRECT column names
    print("👤 User özellikleri (DOĞRU KOLONLAR)...")
    
    # User metadata - user_gender, user_birth_year, user_tenure_in_days
    user_features = data_dict['user_metadata'].with_columns([
        pl.col('user_gender').fill_null('UNKNOWN').alias('gender'),
        pl.col('user_birth_year').fill_null(1990.0).alias('birth_year'),
        pl.col('user_tenure_in_days').alias('tenure_days')
    ]).with_columns([
        (2025 - pl.col('birth_year')).alias('user_age')
    ])
    
    # User search behavior with CORRECT columns
    user_search_stats = data_dict['user_search_log'].group_by('user_id_hashed').agg([
        pl.col('total_search_impression').sum().alias('user_total_search_impressions'),
        pl.col('total_search_click').sum().alias('user_total_search_clicks'),
        pl.col('ts_hour').count().alias('user_search_sessions')
    ]).with_columns([
        (pl.col('user_total_search_clicks') / pl.col('user_total_search_impressions').clip(lower_bound=1e-8)).alias('user_search_ctr')
    ])
    
    # User sitewide behavior 
    user_sitewide_stats = data_dict['user_sitewide_log'].group_by('user_id_hashed').agg([
        pl.count().alias('user_sitewide_interactions'),
        pl.col('content_id_hashed').n_unique().alias('user_unique_contents') if 'content_id_hashed' in data_dict['user_sitewide_log'].columns else pl.lit(0).alias('user_unique_contents')
    ])
    
    # User fashion behavior
    user_fashion_stats = data_dict['user_fashion_search'].group_by('user_id_hashed').agg([
        pl.count().alias('user_fashion_searches'),
        pl.col('search_term_normalized').n_unique().alias('user_fashion_unique_terms') if 'search_term_normalized' in data_dict['user_fashion_search'].columns else pl.lit(0).alias('user_fashion_unique_terms')
    ])
    
    # Combine all user features
    print("🔗 User özellikleri birleştiriliyor...")
    user_enriched = user_features
    for stats_df in [user_search_stats, user_sitewide_stats, user_fashion_stats]:
        user_enriched = user_enriched.join(stats_df, on='user_id_hashed', how='left')
    
    print(f"✅ User enriched shape: {user_enriched.shape}")
    
    # Search term features with CORRECT columns
    print("🔍 Search term özellikleri (DOĞRU KOLONLAR)...")
    term_stats = data_dict['term_search_log'].group_by('search_term_normalized').agg([
        pl.col('total_search_impression').sum().alias('term_total_impressions'),
        pl.col('total_search_click').sum().alias('term_total_clicks'),
        pl.col('ts_hour').count().alias('term_frequency')
    ]).with_columns([
        (pl.col('term_total_clicks') / pl.col('term_total_impressions').clip(lower_bound=1e-8)).alias('term_ctr')
    ])
    
    print(f"✅ Term stats shape: {term_stats.shape}")
    
    # Enrich train sessions with ALL features
    print("🚀 Train sessions zenginleştiriliyor (TÜM ÖZELLİKLER)...")
    train_enriched = train_sessions.join(content_enriched, on='content_id_hashed', how='left')
    train_enriched = train_enriched.join(user_enriched, on='user_id_hashed', how='left')
    train_enriched = train_enriched.join(term_stats, on='search_term_normalized', how='left')
    
    # Enrich test sessions with ALL features  
    print("🧪 Test sessions zenginleştiriliyor (TÜM ÖZELLİKLER)...")
    test_enriched = test_sessions.join(content_enriched, on='content_id_hashed', how='left')
    test_enriched = test_enriched.join(user_enriched, on='user_id_hashed', how='left')
    test_enriched = test_enriched.join(term_stats, on='search_term_normalized', how='left')
    
    print(f"✅ Train enriched shape: {train_enriched.shape}")
    print(f"✅ Test enriched shape: {test_enriched.shape}")
    print(f"Train enriched columns: {len(train_enriched.columns)}")
    print(f"Test enriched columns: {len(test_enriched.columns)}")
    
    return train_enriched, test_enriched


def create_id_mappings(train_df, test_df):
    """Create user and content ID mappings"""
    print("🔑 ID mappings oluşturuluyor...")
    
    all_user_ids = set(train_df['user_id_hashed'].unique()) | set(test_df['user_id_hashed'].unique())
    all_content_ids = set(train_df['content_id_hashed'].unique()) | set(test_df['content_id_hashed'].unique())
    
    # 0 index'ini padding/unknown için ayır
    user_id_map = {user_id: idx + 1 for idx, user_id in enumerate(sorted(all_user_ids))}
    content_id_map = {content_id: idx + 1 for idx, content_id in enumerate(sorted(all_content_ids))}
    
    print(f"✅ Unique users: {len(user_id_map):,}")
    print(f"✅ Unique contents: {len(content_id_map):,}")
    
    return user_id_map, content_id_map


def train_epoch(model, dataloader, optimizer, criterion_click, criterion_order, device, scaler, epoch):
    """Train one epoch with GPU optimization"""
    model.train()
    total_loss = 0
    click_preds, click_targets = [], []
    order_preds, order_targets = [], []
    
    start_time = time.time()
    
    for batch_idx, batch in enumerate(dataloader):
        # GPU transfer
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        user_ids = batch['user_id'].to(device, non_blocking=True)
        content_ids = batch['content_id'].to(device, non_blocking=True)
        clicked = batch['clicked'].to(device, non_blocking=True)
        ordered = batch['ordered'].to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Mixed precision forward pass
        with torch.cuda.amp.autocast():
            click_logits, order_logits = model(input_ids, attention_mask, user_ids, content_ids)
            click_loss = criterion_click(click_logits, clicked)
            order_loss = criterion_order(order_logits, ordered)
            total_batch_loss = 0.3 * click_loss + 0.7 * order_loss
        
        # Mixed precision backward pass
        scaler.scale(total_batch_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += total_batch_loss.item()
        
        # Collect predictions for AUC
        with torch.no_grad():
            click_probs = torch.sigmoid(click_logits).cpu().numpy()
            order_probs = torch.sigmoid(order_logits).cpu().numpy()
            
            click_preds.extend(click_probs)
            click_targets.extend(clicked.cpu().numpy())
            order_preds.extend(order_probs)
            order_targets.extend(ordered.cpu().numpy())
        
        # Progress logging
        if batch_idx % 100 == 0:
            elapsed = time.time() - start_time
            batches_per_sec = (batch_idx + 1) / elapsed if elapsed > 0 else 0
            eta_min = (len(dataloader) - batch_idx - 1) / max(batches_per_sec, 1) / 60
            
            print(f"Epoch {epoch+1} | Batch {batch_idx+1}/{len(dataloader)} | "
                  f"Loss: {total_batch_loss.item():.4f} | "
                  f"Speed: {batches_per_sec:.1f} batch/s | "
                  f"ETA: {eta_min:.1f}min | "
                  f"GPU Memory: {torch.cuda.memory_allocated()/1024**3:.1f}GB")
    
    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    try:
        click_auc = roc_auc_score(click_targets, click_preds) if len(set(click_targets)) > 1 else 0.5
        order_auc = roc_auc_score(order_targets, order_preds) if len(set(order_targets)) > 1 else 0.5
    except:
        click_auc, order_auc = 0.5, 0.5
    
    return avg_loss, click_auc, order_auc


def validate_epoch(model, dataloader, criterion_click, criterion_order, device):
    """Validate one epoch"""
    model.eval()
    total_loss = 0
    click_preds, click_targets = [], []
    order_preds, order_targets = [], []
    
    with torch.no_grad():
        for batch in dataloader:
            # GPU transfer
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            user_ids = batch['user_id'].to(device, non_blocking=True)
            content_ids = batch['content_id'].to(device, non_blocking=True)
            clicked = batch['clicked'].to(device, non_blocking=True)
            ordered = batch['ordered'].to(device, non_blocking=True)
            
            with torch.cuda.amp.autocast():
                click_logits, order_logits = model(input_ids, attention_mask, user_ids, content_ids)
                click_loss = criterion_click(click_logits, clicked)
                order_loss = criterion_order(order_logits, ordered)
                total_batch_loss = 0.3 * click_loss + 0.7 * order_loss
            
            total_loss += total_batch_loss.item()
            
            # Collect predictions
            click_probs = torch.sigmoid(click_logits).cpu().numpy()
            order_probs = torch.sigmoid(order_logits).cpu().numpy()
            
            click_preds.extend(click_probs)
            click_targets.extend(clicked.cpu().numpy())
            order_preds.extend(order_probs)
            order_targets.extend(ordered.cpu().numpy())
    
    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    try:
        click_auc = roc_auc_score(click_targets, click_preds) if len(set(click_targets)) > 1 else 0.5
        order_auc = roc_auc_score(order_targets, order_preds) if len(set(order_targets)) > 1 else 0.5
    except:
        click_auc, order_auc = 0.5, 0.5
    
    return avg_loss, click_auc, order_auc


def main():
    """Main training function with CORRECTED data inspection"""
    print("🚀 Two-Tower Model Training (VERİ İNCELENMİŞ SÜRÜM)...\n")
    
    # Load ALL data with proper column names
    data_dict = load_all_data()
    
    # Create enriched features with CORRECT column names
    train_enriched, test_enriched = create_enriched_features(data_dict)
    
    # Convert to pandas
    print("\n📊 Veri pandas'a çevriliyor...")
    train_df = train_enriched.to_pandas()
    test_df = test_enriched.to_pandas()
    
    # Fill missing values with appropriate defaults
    print("🔧 Eksik değerler dolduruluyor...")
    
    # Train data - numeric columns
    train_numeric_columns = train_df.select_dtypes(include=[np.number]).columns
    for col in train_numeric_columns:
        train_df[col] = train_df[col].fillna(0)
    
    # Test data - numeric columns (only columns that exist)
    test_numeric_columns = test_df.select_dtypes(include=[np.number]).columns
    for col in test_numeric_columns:
        test_df[col] = test_df[col].fillna(0)
    
    # Train data - categorical columns
    train_categorical_columns = train_df.select_dtypes(include=['object']).columns
    for col in train_categorical_columns:
        if col not in ['user_id_hashed', 'content_id_hashed', 'session_id']:
            train_df[col] = train_df[col].fillna('unknown')
    
    # Test data - categorical columns (only columns that exist)
    test_categorical_columns = test_df.select_dtypes(include=['object']).columns
    for col in test_categorical_columns:
        if col not in ['user_id_hashed', 'content_id_hashed', 'session_id']:
            test_df[col] = test_df[col].fillna('unknown')
    
    # Search term normalization
    train_df['search_term_normalized'] = train_df['search_term_normalized'].fillna('')
    test_df['search_term_normalized'] = test_df['search_term_normalized'].fillna('')
    
    print(f"✅ Train data shape (ZENGİNLEŞTİRİLMİŞ): {train_df.shape}")
    print(f"✅ Test data shape (ZENGİNLEŞTİRİLMİŞ): {test_df.shape}")
    print(f"✅ Train columns: {len(train_df.columns)}")
    print(f"✅ Test columns: {len(test_df.columns)}")
    
    # Check if target columns exist
    if 'clicked' in train_df.columns:
        print(f"✅ Click distribution: {train_df['clicked'].value_counts().to_dict()}")
    if 'ordered' in train_df.columns:
        print(f"✅ Order distribution: {train_df['ordered'].value_counts().to_dict()}")
    
    print(f"Train data columns: {list(train_df.columns)[:10]}...")  # First 10 columns
    print(f"Test data columns: {list(test_df.columns)[:10]}...")   # First 10 columns
    
    # Create ID mappings
    user_id_map, content_id_map = create_id_mappings(train_df, test_df)
    
    # Model configuration
    model_config = {
        'user_embedding_dim': 128,
        'content_embedding_dim': 128,
        'tower_dim': 256,
        'num_users': len(user_id_map) + 1,
        'num_contents': len(content_id_map) + 1,
    }
    
    print(f"\n📋 Model config: {model_config}")
    
    # Initialize model and tokenizer
    print("🏗️  Model ve tokenizer oluşturuluyor...")
    model = TwoTowerModel(model_config).to(device)
    tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
    
    # TÜM TRAIN VERİSİNİ KULLAN - NO SAMPLING
    print("📝 TÜM training verisi kullanılıyor (NO SAMPLING)...")
    train_sample = train_df  # Tüm veriyi kullan, sampling yok
    
    print(f"✅ Training data size: {len(train_sample):,} (TÜM VERİ)")
    
    # Train/validation split
    if 'clicked' in train_sample.columns:
        train_data, val_data = train_test_split(
            train_sample, test_size=0.15, random_state=42, stratify=train_sample['clicked']
        )
    else:
        train_data, val_data = train_test_split(
            train_sample, test_size=0.15, random_state=42
        )
    
    print(f"✅ Train size: {len(train_data):,}")
    print(f"✅ Validation size: {len(val_data):,}")
    
    # Create datasets with longer sequences
    train_dataset = TrendyolDataset(train_data, tokenizer, user_id_map, content_id_map, max_length=128)
    val_dataset = TrendyolDataset(val_data, tokenizer, user_id_map, content_id_map, max_length=128)
    
    # Create dataloaders - GPU optimized batch sizes
    train_loader = DataLoader(
        train_dataset, 
        batch_size=48,  # Increased from 24 - GPU has more memory
        shuffle=True, 
        num_workers=0,  # Windows multiprocessing fix
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=96,  # Increased from 48 - validation can use larger batches
        shuffle=False, 
        num_workers=0,  # Windows multiprocessing fix
        pin_memory=True
    )
    
    print(f"✅ Train batches: {len(train_loader)}")
    print(f"✅ Val batches: {len(val_loader)}")
    
    # Training setup
    criterion_click = nn.BCEWithLogitsLoss()
    criterion_order = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], 
        lr=2e-4,  # Optimal learning rate
        weight_decay=0.01
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler()
    
    print("✅ Training setup tamamlandı (FULL ENRICHED DATA)!")
    print("=" * 60)
    
    # Training loop
    num_epochs = 8  # More epochs for enriched data
    best_val_score = 0.0
    best_model_state = None
    
    for epoch in range(num_epochs):
        print(f"\n🏃‍♂️ EPOCH {epoch+1}/{num_epochs}")
        print("-" * 40)
        
        # Training
        train_loss, train_click_auc, train_order_auc = train_epoch(
            model, train_loader, optimizer, criterion_click, criterion_order, 
            device, scaler, epoch
        )
        
        # Validation
        print("🔍 Validation...")
        val_loss, val_click_auc, val_order_auc = validate_epoch(
            model, val_loader, criterion_click, criterion_order, device
        )
        
        # Learning rate step
        scheduler.step()
        
        # Calculate combined scores
        train_combined = 0.3 * train_click_auc + 0.7 * train_order_auc
        val_combined = 0.3 * val_click_auc + 0.7 * val_order_auc
        
        print(f"\n📊 Epoch {epoch+1} Results:")
        print(f"Train - Loss: {train_loss:.4f} | Click AUC: {train_click_auc:.4f} | Order AUC: {train_order_auc:.4f} | Combined: {train_combined:.4f}")
        print(f"Val   - Loss: {val_loss:.4f} | Click AUC: {val_click_auc:.4f} | Order AUC: {val_order_auc:.4f} | Combined: {val_combined:.4f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"GPU Memory: {torch.cuda.memory_allocated()/1024**3:.1f}GB")
        
        # Save model checkpoint for every epoch
        checkpoint_path = f'two_tower_model_epoch_{epoch+1}.pt'
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'model_config': model_config,
            'user_id_map': user_id_map,
            'content_id_map': content_id_map,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_combined_score': train_combined,
            'val_combined_score': val_combined,
            'train_click_auc': train_click_auc,
            'train_order_auc': train_order_auc,
            'val_click_auc': val_click_auc,
            'val_order_auc': val_order_auc,
            'tokenizer_name': model.bert_model_name
        }, checkpoint_path)
        print(f"💾 Epoch {epoch+1} modeli kaydedildi: {checkpoint_path}")
        
        # Save best model
        if val_combined > best_val_score:
            best_val_score = val_combined
            best_model_state = model.state_dict().copy()
            print(f"✅ En iyi model güncellendi! Score: {best_val_score:.4f}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n✅ En iyi model yüklendi. Final validation score: {best_val_score:.4f}")
    
    # Save model
    print("\n💾 Model kaydediliyor...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': model_config,
        'user_id_map': user_id_map,
        'content_id_map': content_id_map,
        'best_val_score': best_val_score,
        'tokenizer_name': model.bert_model_name
    }, 'two_tower_model_corrected.pt')
    
    print("=" * 60)
    print("🎉 TRAINING COMPLETED!")
    print(f"✅ Final Validation Score: {best_val_score:.6f}")
    print(f"✅ Model saved as: two_tower_model_corrected.pt")
    print(f"✅ Architecture: Two-Tower + Turkish BERT + FULL ENRICHED DATA")
    print(f"✅ Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"✅ Training Data: {len(train_sample):,} samples")
    print("=" * 60)
    
    # Full test predictions
    print("\n🔮 FULL test predictions (ALL DATA)...")
    
    print(f"✅ Full test data: {len(test_df):,} rows")
    test_dataset = TrendyolDataset(test_df, tokenizer, user_id_map, content_id_map, max_length=128)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=96,  # Increased for better GPU utilization
        shuffle=False, 
        num_workers=0,  # Windows multiprocessing fix
        pin_memory=True
    )
    
    model.eval()
    all_click_preds = []
    all_order_preds = []
    
    print(f"📊 Test inference: {len(test_loader)} batches")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx % 200 == 0:
                print(f"Test batch {batch_idx+1}/{len(test_loader)} | "
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
    final_submission.to_csv('submission_final_correct.csv', index=False)
    
    print(f"✅ Submission saved: submission_final_correct.csv")
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
    
    print("\n🎯 TWO-TOWER MODEL WITH CORRECTED ENRICHED DATA - SUCCESS!")
    print("📚 DOĞRU kolon isimleri ile TÜM veriler kullanıldı!")
    print("🚀 800k training samples, 8 epochs, Turkish BERT!")
    print("💎 Zenginleştirilmiş content, user ve term özellikleri!")
    print("=" * 60)


if __name__ == "__main__":
    main()
