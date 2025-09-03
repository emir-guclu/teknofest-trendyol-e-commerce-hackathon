# Veri_ekleme.py
# Bu script, ham verileri okur, yeni özellikler üretir ve son zenginleştirilmiş
# train ve test setlerini kaydeder.

import pandas as pd
import numpy as np
from pathlib import Path
import os

print("Tüm özellik mühendisliği adımları başlıyor...")

# --- 1. Ham Verileri Yükleme ---
print("[1/5] Ham veri dosyaları okunuyor...")
try:
    df_train_sessions = pd.read_parquet('CatBoost/enriched_train_data_v2_plus.parquet')
    df_test_sessions = pd.read_parquet('CatBoost/enriched_test_data_v2_plus.parquet')

    df_content_meta = pd.read_parquet('data/content/metadata.parquet')
    df_price = pd.read_parquet('data/content/price_rate_review_data.parquet')
    df_user_meta = pd.read_parquet('data/user/metadata.parquet')

    # Yeni özellik için gerekli log dosyası
    df_interaction_logs = pd.read_parquet('data/user/fashion_sitewide_log.parquet')
except FileNotFoundError as e:
    print(f"HATA: Dosya bulunamadı! Lütfen dosya yolunu kontrol edin: {e}")
    exit()

# --- 2. Ürün ve Kullanıcı Meta Özelliklerini Oluşturma ---
print("[2/5] Ürün yaşı, indirim oranı ve kullanıcı yaşı özellikleri oluşturuluyor...")

# Veri setindeki "şimdiki zamanı" bul
present_time = pd.concat([df_train_sessions['ts_hour'], df_test_sessions['ts_hour']]).max()

# Ürün yaşını gün olarak hesapla
df_content_meta['product_age_days'] = (present_time - df_content_meta['content_creation_date']).dt.days

# İndirim oranını hesapla
df_price['original_price'] = np.where(df_price['original_price'] < df_price['selling_price'], df_price['selling_price'], df_price['original_price'])
df_price['discount_percentage'] = np.where(df_price['original_price'] > 0, (df_price['original_price'] - df_price['selling_price']) / df_price['original_price'], 0) * 100
# Her ürün için en güncel fiyat bilgisini al
df_price = df_price.sort_values('update_date', ascending=False).drop_duplicates('content_id_hashed')

# Kullanıcı yaşını hesapla
present_year = present_time.year
df_user_meta['user_age'] = present_year - df_user_meta['user_birth_year']

# Ana dataframe'lere eklemek için özellikleri hazırla
df_product_age = df_content_meta[['content_id_hashed', 'product_age_days']]
df_discount = df_price[['content_id_hashed', 'discount_percentage']]
df_user_age = df_user_meta[['user_id_hashed', 'user_age']]

# --- 3. Meta Özellikleri Ana Veri Setlerine Ekleme ---
print("[3/5] Meta özellikler ana veri setlerine birleştiriliyor...")
def enrich_with_metadata(df, df_product_age, df_discount, df_user_age):
    df_enriched = pd.merge(df, df_product_age, on='content_id_hashed', how='left')
    df_enriched = pd.merge(df_enriched, df_discount, on='content_id_hashed', how='left')
    df_enriched = pd.merge(df_enriched, df_user_age, on='user_id_hashed', how='left')
    
    # NaN değerleri doldur
    df_enriched['product_age_days'].fillna(0, inplace=True)
    df_enriched['discount_percentage'].fillna(0, inplace=True)
    df_enriched['user_age'].fillna(-1, inplace=True)
    return df_enriched

train_df_v2 = enrich_with_metadata(df_train_sessions, df_product_age, df_discount, df_user_age)
test_df_v2 = enrich_with_metadata(df_test_sessions, df_product_age, df_discount, df_user_age)
print("Meta özellikler eklendi.")


# --- 4. Kullanıcı-İçerik Geçmiş Etkileşimlerini Güvenle Ekleme ---
print("[4/5] Geçmiş kullanıcı-içerik etkileşim özellikleri ekleniyor (merge_asof)...")
def add_user_content_interaction_features(df_sessions, df_interaction_logs):
    logs = df_interaction_logs[['user_id_hashed', 'content_id_hashed', 'ts_hour', 'total_click']].copy()
    sessions = df_sessions.copy()
    
    logs['ts_hour'] = pd.to_datetime(logs['ts_hour'])
    sessions['ts_hour'] = pd.to_datetime(sessions['ts_hour'])
    
    logs = logs.sort_values('ts_hour')
    # total_click 0'dan büyükse 1, değilse 0 alarak tıklama olayını sayalım
    logs['click_event'] = (logs['total_click'] > 0).astype(int)
    logs['user_content_past_click_count'] = logs.groupby(['user_id_hashed', 'content_id_hashed'])['click_event'].cumsum()
    
    sessions = sessions.sort_values('ts_hour')
    
    df_enriched = pd.merge_asof(
        left=sessions,
        right=logs[['user_id_hashed', 'content_id_hashed', 'ts_hour', 'user_content_past_click_count']],
        on='ts_hour',
        by=['user_id_hashed', 'content_id_hashed'],
        direction='backward'
    )
    
    df_enriched['user_content_past_click_count'].fillna(0, inplace=True)
    return df_enriched

# Fonksiyonu çağırarak son özelliği de ekleyelim
train_df_v3 = add_user_content_interaction_features(train_df_v2, df_interaction_logs)
test_df_v3 = add_user_content_interaction_features(test_df_v2, df_interaction_logs)
print("Geçmiş etkileşim özellikleri eklendi.")


# --- 5. Sonuçları Kaydetme ---
print("[5/5] Zenginleştirilmiş son veri setleri kaydediliyor...")

# Nihai dosya adları (v3 veya istediğiniz bir versiyon adı)
output_path = "YeniDeneme/veri/"
train_output_path = output_path + 'train_sessions.parquet'
test_output_path = output_path + 'test_sessions.parquet'

train_df_v3.to_parquet(train_output_path, index=False)
test_df_v3.to_parquet(test_output_path, index=False)

print(f"\nİşlem tamamlandı! 🚀")
print(f"Train verisi kaydedildi: {train_output_path}")
print(f"Test verisi kaydedildi: {test_output_path}")

print("\nTrain setinin son hali (ilk 5 satır):")
print(train_df_v3.head())
print("\nTrain setinin bilgisi:")
print(train_df_v3.info())