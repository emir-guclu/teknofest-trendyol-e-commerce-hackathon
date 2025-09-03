# Veri_ekleme.py (v4 - sess_price_z eklenmiş hali)
# Bu script, ham verileri okur, yeni özellikler üretir ve son zenginleştirilmiş
# train ve test setlerini kaydeder.

import pandas as pd
import numpy as np
from pathlib import Path
import time
print("Tüm özellik mühendisliği adımları başlıyor...")


# --- 1. Ham Verileri Yükleme ---
print("[1/6] Ham veri dosyaları okunuyor...")
root = Path("C:/Projects/Veri/YeniDeneme")

try:
    df_train_sessions = pd.read_parquet('CatBoost/enriched_train_data_v2_plus.parquet')
    df_test_sessions = pd.read_parquet('CatBoost/enriched_test_data_v2_plus.parquet')

    df_content_meta = pd.read_parquet('data/content/metadata.parquet')
    df_price = pd.read_parquet('data/content/price_rate_review_data.parquet')
    df_user_meta = pd.read_parquet('data/user/metadata.parquet')

    df_interaction_logs = pd.read_parquet('data/user/fashion_sitewide_log.parquet')
except FileNotFoundError as e:
    print(f"HATA: Dosya bulunamadı! Lütfen dosya yolunu kontrol edin: {e}")
    exit()


# --- 2. Ürün ve Kullanıcı Meta Özelliklerini Oluşturma ---
print("[2/6] Ürün yaşı, indirim oranı ve kullanıcı yaşı özellikleri oluşturuluyor...")
present_time = pd.concat([df_train_sessions['ts_hour'], df_test_sessions['ts_hour']]).max()
df_content_meta['product_age_days'] = (present_time - df_content_meta['content_creation_date']).dt.days
df_price['original_price'] = np.where(df_price['original_price'] < df_price['selling_price'], df_price['selling_price'], df_price['original_price'])
df_price['discount_percentage'] = np.where(df_price['original_price'] > 0, (df_price['original_price'] - df_price['selling_price']) / df_price['original_price'], 0) * 100
df_price = df_price.sort_values('update_date', ascending=False).drop_duplicates('content_id_hashed')
present_year = present_time.year
df_user_meta['user_age'] = present_year - df_user_meta['user_birth_year']

# Meta özellikleri birleştirmeye hazırla
df_product_age = df_content_meta[['content_id_hashed', 'product_age_days']]
# selling_price'ı da ana df'e eklemek için alalım
df_discount_and_price = df_price[['content_id_hashed', 'discount_percentage', 'selling_price']]
df_user_age = df_user_meta[['user_id_hashed', 'user_age']]

# --- 3. Meta Özellikleri Ana Veri Setlerine Ekleme ---
print("[3/6] Meta özellikler ana veri setlerine birleştiriliyor...")
def enrich_with_metadata(df, df_product_age, df_discount_and_price, df_user_age):

    df_enriched = df.copy() # Orijinal df'i kopyalayarak başlayalım

    df_enriched = pd.merge(df, df_product_age, on='content_id_hashed', how='left')
    df_enriched = pd.merge(df_enriched, df_discount_and_price, on='content_id_hashed', how='left')
    df_enriched = pd.merge(df_enriched, df_user_age, on='user_id_hashed', how='left')

        # --- THE DEFINITIVE FIX IS HERE ---
    # 1. Drop the old price column ('selling_price_x') that came from the base file.
    if 'selling_price_x' in df_enriched.columns:
        df_enriched.drop(columns=['selling_price_x'], inplace=True)
        
    # 2. Rename the new, up-to-date price column ('selling_price_y') to our desired name.
    if 'selling_price_y' in df_enriched.columns:
        df_enriched.rename(columns={'selling_price_y': 'selling_price'}, inplace=True)
    # --- FIX ENDS HERE ---


    df_enriched['product_age_days'].fillna(0, inplace=True)
    df_enriched['discount_percentage'].fillna(0, inplace=True)
    df_enriched['user_age'].fillna(-1, inplace=True)

    # selling_price'da NaN kalırsa (eşleşmeyen ürün), ortalama ile doldurabiliriz
    df_enriched['selling_price'].fillna(df_enriched['selling_price'].mean(), inplace=True)
    return df_enriched

train_df_v2 = enrich_with_metadata(df_train_sessions, df_product_age, df_discount_and_price, df_user_age)
test_df_v2 = enrich_with_metadata(df_test_sessions, df_product_age, df_discount_and_price, df_user_age)
print("Meta özellikler eklendi.")


# --- 4. Kullanıcı-İçerik Geçmiş Etkileşimlerini Güvenle Ekleme ---
print("[4/6] Geçmiş kullanıcı-içerik etkileşim özellikleri ekleniyor (merge_asof)...")
def add_user_content_interaction_features(df_sessions, df_interaction_logs):
    # Bu fonksiyonun içi önceki cevapta olduğu gibi aynı kalıyor...
    logs = df_interaction_logs[['user_id_hashed', 'content_id_hashed', 'ts_hour', 'total_click']].copy()
    sessions = df_sessions.copy()
    logs['ts_hour'] = pd.to_datetime(logs['ts_hour']); sessions['ts_hour'] = pd.to_datetime(sessions['ts_hour'])
    logs = logs.sort_values('ts_hour')
    logs['click_event'] = (logs['total_click'] > 0).astype(int)
    logs['user_content_past_click_count'] = logs.groupby(['user_id_hashed', 'content_id_hashed'])['click_event'].cumsum()
    sessions = sessions.sort_values('ts_hour')
    df_enriched = pd.merge_asof(
        left=sessions,
        right=logs[['user_id_hashed', 'content_id_hashed', 'ts_hour', 'user_content_past_click_count']],
        on='ts_hour', by=['user_id_hashed', 'content_id_hashed'], direction='backward'
    )
    df_enriched['user_content_past_click_count'].fillna(0, inplace=True)
    return df_enriched

train_df_v3 = add_user_content_interaction_features(train_df_v2, df_interaction_logs)
test_df_v3 = add_user_content_interaction_features(test_df_v2, df_interaction_logs)
print("Geçmiş etkileşim özellikleri eklendi.")

# <<< YENİ EKLENEN KISIM BAŞLANGICI >>>
# --- 5. Oturum İçi Fiyat Bağlam Özelliği Ekleme ---
print("[5/6] Oturum içi fiyat bağlam özellikleri (sess_price_z) ekleniyor...")
def add_session_price_features(df):
    if 'selling_price' not in df.columns:
        raise ValueError("'selling_price' sütunu DataFrame'de bulunamadı!")
    df['sess_price_mean'] = df.groupby('session_id')['selling_price'].transform('mean')
    df['sess_price_std'] = df.groupby('session_id')['selling_price'].transform('std')
    df['sess_price_z'] = np.where(df['sess_price_std'] > 0, (df['selling_price'] - df['sess_price_mean']) / df['sess_price_std'], 0)
    df.drop(columns=['sess_price_mean', 'sess_price_std'], inplace=True)
    df['sess_price_z'].fillna(0, inplace=True)
    return df

train_df_v4 = add_session_price_features(train_df_v3)
test_df_v4 = add_session_price_features(test_df_v3)
print("Oturum içi fiyat özellikleri eklendi.")
# <<< YENİ EKLENEN KISIM SONU >>>


# --- 6. Sonuçları Kaydetme ---
print("[6/6] Zenginleştirilmiş son veri setleri kaydediliyor...")

train_output_path = root / 'veri/train_sessions_v2.parquet'
test_output_path = root / 'veri/test_sessions_v2.parquet'

train_df_v4.to_parquet(train_output_path, index=False)
test_df_v4.to_parquet(test_output_path, index=False)

print(f"\nİşlem tamamlandı! 🚀")
print(f"Train verisi kaydedildi: {train_output_path}")
print(f"Test verisi kaydedildi: {test_output_path}")

print("\nTrain setinin son hali (yeni eklenen özelliklerle birlikte ilk 5 satır):")
print(train_df_v4[['session_id', 'content_id_hashed', 'selling_price', 'sess_price_z']].head())
print("\nTrain setinin bilgisi:")
print(train_df_v4.info())