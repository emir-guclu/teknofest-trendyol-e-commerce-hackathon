#!/usr/bin/env python3
"""
Veri yapısını kontrol et
"""

import polars as pl

DATA_PATH = "trendyol-e-ticaret-hackathonu-2025-kaggle/data/"

# Content verilerini kontrol et
print("=== CONTENT DATA COLUMNS ===")
content_metadata = pl.read_parquet(f"{DATA_PATH}content/metadata.parquet")
print(f"Content metadata columns: {content_metadata.columns}")
print(content_metadata.head(2))

content_price = pl.read_parquet(f"{DATA_PATH}content/price_rate_review_data.parquet")
print(f"\nContent price columns: {content_price.columns}")
print(content_price.head(2))

content_search = pl.read_parquet(f"{DATA_PATH}content/search_log.parquet")
print(f"\nContent search columns: {content_search.columns}")
print(content_search.head(2))

content_sitewide = pl.read_parquet(f"{DATA_PATH}content/sitewide_log.parquet")
print(f"\nContent sitewide columns: {content_sitewide.columns}")
print(content_sitewide.head(2))

# User verilerini kontrol et
print("\n=== USER DATA COLUMNS ===")
user_metadata = pl.read_parquet(f"{DATA_PATH}user/metadata.parquet")
print(f"User metadata columns: {user_metadata.columns}")

user_search = pl.read_parquet(f"{DATA_PATH}user/search_log.parquet")
print(f"User search columns: {user_search.columns}")

user_sitewide = pl.read_parquet(f"{DATA_PATH}user/sitewide_log.parquet")
print(f"User sitewide columns: {user_sitewide.columns}")

# Term verilerini kontrol et
print("\n=== TERM DATA COLUMNS ===")
term_search = pl.read_parquet(f"{DATA_PATH}term/search_log.parquet")
print(f"Term search columns: {term_search.columns}")

print("\n=== SESSIONS DATA ===")
train_sessions = pl.read_parquet(f"{DATA_PATH}train_sessions.parquet")
print(f"Train sessions columns: {train_sessions.columns}")

test_sessions = pl.read_parquet(f"{DATA_PATH}test_sessions.parquet")  
print(f"Test sessions columns: {test_sessions.columns}")
