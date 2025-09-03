"""
Script to prepare enriched train and test session datasets by merging session tables with aggregated
user and content features. Produces two parquet files:
  - processed_train_sessions.parquet
  - processed_test_sessions.parquet
in the data folder.

Usage:
  python tools/prepare_sessions.py

Requires:
  pandas, pyarrow
"""
import pandas as pd
from pathlib import Path

# Define data paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "trendyol-e-ticaret-hackathonu-2025-kaggle" / "data"
OUTPUT_TRAIN = DATA_DIR / "processed_train_sessions.parquet"
OUTPUT_TEST = DATA_DIR / "processed_test_sessions.parquet"

# Session files
TRAIN_SESSIONS = DATA_DIR / "train_sessions.parquet"
TEST_SESSIONS = DATA_DIR / "test_sessions.parquet"

# User tables
USER_DIR = DATA_DIR / "user"
USER_TABLES = {
    'metadata': USER_DIR / 'metadata.parquet',
    'search_log': USER_DIR / 'search_log.parquet',
    'sitewide_log': USER_DIR / 'sitewide_log.parquet',
    'top_terms': USER_DIR / 'top_terms_log.parquet',
    'fashion_search': USER_DIR / 'fashion_search_log.parquet',
    'fashion_sitewide': USER_DIR / 'fashion_sitewide_log.parquet',
}

# Content tables
CONTENT_DIR = DATA_DIR / "content"
CONTENT_TABLES = {
    'metadata': CONTENT_DIR / 'metadata.parquet',
    'price_review': CONTENT_DIR / 'price_rate_review_data.parquet',
    'search_log': CONTENT_DIR / 'search_log.parquet',
    'sitewide_log': CONTENT_DIR / 'sitewide_log.parquet',
    'top_terms': CONTENT_DIR / 'top_terms_log.parquet',
}


def aggregate_user_features():
    """Aggregate user-level features from multiple user tables"""
    # Load metadata
    df_meta = pd.read_parquet(USER_TABLES['metadata'])
    user_df = df_meta.groupby('user_id_hashed').agg({
        'user_birth_year': 'mean',
        'user_tenure_in_days': 'mean'
    }).rename(columns={
        'user_birth_year': 'user_birth_year_mean',
        'user_tenure_in_days': 'user_tenure_days_mean'
    })
    # Add gender as most frequent
    gender_mode = df_meta.groupby('user_id_hashed')['user_gender'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
    user_df['user_gender_mode'] = gender_mode

    # Aggregations for logs
    # search_log
    df = pd.read_parquet(USER_TABLES['search_log'])
    agg = df.groupby('user_id_hashed').agg({
        'total_search_click': ['sum', 'mean'],
        'total_search_impression': ['sum', 'mean'],
        'ts_hour': 'nunique'
    })
    agg.columns = [f'user_search_{c[0]}_{c[1]}' for c in agg.columns]
    user_df = user_df.join(agg, how='left')

    # sitewide_log
    df = pd.read_parquet(USER_TABLES['sitewide_log'])
    agg = df.groupby('user_id_hashed').agg({
        'total_cart': ['sum', 'mean'],
        'total_click': ['sum', 'mean'],
        'total_fav': ['sum', 'mean'],
        'total_order': ['sum', 'mean'],
        'ts_hour': 'nunique'
    })
    agg.columns = [f'user_sitewide_{c[0]}_{c[1]}' for c in agg.columns]
    user_df = user_df.join(agg, how='left')

    # top_terms_log
    df = pd.read_parquet(USER_TABLES['top_terms'])
    agg = df.groupby('user_id_hashed').agg({
        'total_search_click': ['sum'],
        'total_search_impression': ['sum'],
        'search_term_normalized': 'nunique',
        'ts_hour': 'nunique'
    })
    agg.columns = [f'user_topterms_{c[0]}_{c[1]}' for c in agg.columns]
    user_df = user_df.join(agg, how='left')

    # fashion_search_log
    df = pd.read_parquet(USER_TABLES['fashion_search'])
    agg = df.groupby('user_id_hashed').agg({
        'total_search_click': ['sum'],
        'total_search_impression': ['sum'],
        'ts_hour': 'nunique'
    })
    agg.columns = [f'user_fashsearch_{c[0]}_{c[1]}' for c in agg.columns]
    user_df = user_df.join(agg, how='left')

    # fashion_sitewide_log
    df = pd.read_parquet(USER_TABLES['fashion_sitewide'])
    agg = df.groupby('user_id_hashed').agg({
        'total_cart': ['sum'],
        'total_click': ['sum'],
        'total_fav': ['sum'],
        'total_order': ['sum'],
        'ts_hour': 'nunique'
    })
    agg.columns = [f'user_fashsite_{c[0]}_{c[1]}' for c in agg.columns]
    user_df = user_df.join(agg, how='left')

    return user_df.reset_index()


def aggregate_content_features():
    """Aggregate content-level features from multiple content tables"""
    # metadata
    df_meta = pd.read_parquet(CONTENT_TABLES['metadata'])
    content_df = df_meta.set_index('content_id_hashed')

    # price and review
    df = pd.read_parquet(CONTENT_TABLES['price_review'])
    agg = df.groupby('content_id_hashed').agg({
        'content_rate_avg': 'mean',
        'content_rate_count': 'sum',
        'content_review_count': 'sum',
        'content_review_wth_media_count': 'sum',
        'discounted_price': 'mean',
        'original_price': 'mean',
        'selling_price': 'mean'
    }).rename(columns=lambda x: f'cont_pr_{x}')
    content_df = content_df.join(agg, how='left')

    # content search_log
    df = pd.read_parquet(CONTENT_TABLES['search_log'])
    agg = df.groupby('content_id_hashed').agg({
        'total_search_click': ['sum', 'mean'],
        'total_search_impression': ['sum', 'mean'],
        'date': 'nunique'
    })
    agg.columns = [f'cont_slog_{c[0]}_{c[1]}' for c in agg.columns]
    content_df = content_df.join(agg, how='left')

    # content sitewide_log
    df = pd.read_parquet(CONTENT_TABLES['sitewide_log'])
    agg = df.groupby('content_id_hashed').agg({
        'total_cart': ['sum', 'mean'],
        'total_click': ['sum', 'mean'],
        'total_fav': ['sum', 'mean'],
        'total_order': ['sum', 'mean'],
        'date': 'nunique'
    })
    agg.columns = [f'cont_sw_{c[0]}_{c[1]}' for c in agg.columns]
    content_df = content_df.join(agg, how='left')

    # top_terms_log
    df = pd.read_parquet(CONTENT_TABLES['top_terms'])
    agg = df.groupby('content_id_hashed').agg({
        'total_search_click': ['sum'],
        'total_search_impression': ['sum'],
        'search_term_normalized': 'nunique',
        'date': 'nunique'
    })
    agg.columns = [f'cont_top_{c[0]}_{c[1]}' for c in agg.columns]
    content_df = content_df.join(agg, how='left')

    return content_df.reset_index()


def prepare_and_save():
    # Find common columns
    train = pd.read_parquet(TRAIN_SESSIONS)
    test = pd.read_parquet(TEST_SESSIONS)
    common_cols = ['session_id', 'user_id_hashed', 'content_id_hashed', 'search_term_normalized', 'ts_hour']

    # Labels in train
    labels = ['clicked', 'added_to_cart', 'added_to_fav', 'ordered']

    # Base frames
    train_base = train[common_cols + labels]
    test_base = test[common_cols]

    # Aggregated features
    user_feats = aggregate_user_features()
    content_feats = aggregate_content_features()

    # Merge
    train_enriched = train_base.merge(user_feats, on='user_id_hashed', how='left')
    train_enriched = train_enriched.merge(content_feats, on='content_id_hashed', how='left')
    test_enriched = test_base.merge(user_feats, on='user_id_hashed', how='left')
    test_enriched = test_enriched.merge(content_feats, on='content_id_hashed', how='left')

    # Save
    train_enriched.to_parquet(OUTPUT_TRAIN, index=False)
    test_enriched.to_parquet(OUTPUT_TEST, index=False)
    print(f"Saved processed train to {OUTPUT_TRAIN}")
    print(f"Saved processed test to {OUTPUT_TEST}")


if __name__ == '__main__':
    prepare_and_save()
