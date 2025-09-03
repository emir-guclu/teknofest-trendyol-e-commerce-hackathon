import pandas as pd
import numpy as np
import lightgbm as lgb
import polars as pl
import warnings
warnings.filterwarnings('ignore')

# --- 1. YARDIMCI FONKSİYONLAR ---
def load_data(data_path):
    """Gerekli tüm verileri yükler."""
    train_sessions = pl.read_parquet(f"{data_path}/train_sessions.parquet")
    test_sessions = pl.read_parquet(f"{data_path}/test_sessions.parquet")
    user_metadata = pl.read_parquet(f"{data_path}/user/metadata.parquet")
    user_sitewide_log = pl.read_parquet(f"{data_path}/user/sitewide_log.parquet")
    user_search_log = pl.read_parquet(f"{data_path}/user/search_log.parquet")
    content_metadata = pl.read_parquet(f"{data_path}/content/metadata.parquet")
    content_price_rate_review = pl.read_parquet(f"{data_path}/content/price_rate_review_data.parquet")
    content_search_log = pl.read_parquet(f"{data_path}/content/search_log.parquet")
    content_sitewide_log = pl.read_parquet(f"{data_path}/content/sitewide_log.parquet")
    return {
        'train_sessions': train_sessions,
        'test_sessions': test_sessions,
        'user_metadata': user_metadata,
        'user_sitewide_log': user_sitewide_log,
        'user_search_log': user_search_log,
        'content_metadata': content_metadata,
        'content_price_rate_review': content_price_rate_review,
        'content_search_log': content_search_log,
        'content_sitewide_log': content_sitewide_log
    }

def filter_temporal_data(log_df, session_df, key, time_col, log_time_col=None):
    """Veri sızıntısını önlemek için logları session zamanından öncesiyle sınırlar."""
    if log_time_col is None:
        log_time_col = time_col
    
    session_times = session_df.select([pl.col(key), pl.col(time_col)]).unique()
    
    # Log verilerinde zaman sütunu kontrol et
    if log_time_col not in log_df.columns:
        print(f"Warning: {log_time_col} not found in log data. Available columns: {log_df.columns}")
        return log_df  # Filtreleme yapmadan döndür
    
    filtered_log = (
        log_df
        .join(session_times, on=key, how="inner", suffix="_session")
        .filter(pl.col(log_time_col) < pl.col(f"{time_col}_session"))
        .drop(f"{time_col}_session")
    )
    return filtered_log

def create_user_features(user_metadata, user_sitewide_log, user_search_log, session_df):
    user_sitewide_filtered = filter_temporal_data(user_sitewide_log, session_df, "user_id_hashed", "ts_hour")
    user_search_filtered = filter_temporal_data(user_search_log, session_df, "user_id_hashed", "ts_hour")
    user_features = user_metadata.with_columns([
        (2025 - pl.col("user_birth_year")).alias("user_age"),
        pl.col("user_gender").fill_null("UNKNOWN").alias("user_gender_clean"),
        pl.when(pl.col("user_tenure_in_days") <= 30).then(pl.lit("new"))
        .when(pl.col("user_tenure_in_days") <= 365).then(pl.lit("regular"))
        .otherwise(pl.lit("loyal")).alias("user_tenure_category")
    ])
    if user_sitewide_filtered.height > 0:
        user_sitewide_agg = (
            user_sitewide_filtered
            .group_by("user_id_hashed")
            .agg([
                pl.col("total_click").sum().alias("user_total_clicks"),
                pl.col("total_cart").sum().alias("user_total_cart"),
                pl.col("total_fav").sum().alias("user_total_fav"),
                pl.col("total_order").sum().alias("user_total_orders"),
                pl.col("total_click").mean().alias("user_avg_clicks"),
                pl.col("ts_hour").count().alias("user_activity_days")
            ])
        )
        user_features = user_features.join(user_sitewide_agg, on="user_id_hashed", how="left")
    if user_search_filtered.height > 0:
        user_search_agg = (
            user_search_filtered
            .group_by("user_id_hashed")
            .agg([
                pl.col("total_search_impression").sum().alias("user_search_impressions"),
                pl.col("total_search_click").sum().alias("user_search_clicks"),
                pl.col("total_search_impression").count().alias("user_search_days"),
                (pl.col("total_search_click").sum() / pl.col("total_search_impression").sum()).alias("user_search_ctr")
            ])
        )
        user_features = user_features.join(user_search_agg, on="user_id_hashed", how="left")
    numeric_cols = [col for col in user_features.columns if col.startswith("user_") and col != "user_id_hashed" and col != "user_gender_clean" and col != "user_tenure_category"]
    user_features = user_features.with_columns([
        pl.col(col).fill_null(0) for col in numeric_cols
    ])
    return user_features

def create_content_features(content_metadata, content_price_rate_review, content_search_log, content_sitewide_log, session_df):
    # Session'larda ts_hour sütunu var, content log'larda farklı zaman sütunları olabilir
    content_search_time_col = "ts_hour" if "ts_hour" in content_search_log.columns else "date"
    content_sitewide_time_col = "ts_hour" if "ts_hour" in content_sitewide_log.columns else "date"
    
    content_search_filtered = filter_temporal_data(content_search_log, session_df, "content_id_hashed", 
                                                 "ts_hour", content_search_time_col)
    content_sitewide_filtered = filter_temporal_data(content_sitewide_log, session_df, "content_id_hashed", 
                                                   "ts_hour", content_sitewide_time_col)
    content_features = content_metadata.with_columns([
        pl.col("level1_category_name").fill_null("UNKNOWN").alias("level1_category_clean"),
        pl.col("level2_category_name").fill_null("UNKNOWN").alias("level2_category_clean"),
        pl.col("leaf_category_name").fill_null("UNKNOWN").alias("leaf_category_clean"),
        (pl.date(2025, 7, 12) - pl.col("content_creation_date").dt.date()).dt.total_days().alias("content_age_days"),
        pl.col("cv_tags").is_not_null().alias("has_cv_tags"),
        pl.col("attribute_type_count").fill_null(0),
        pl.col("total_attribute_option_count").fill_null(0),
        pl.col("merchant_count").fill_null(1),
        pl.col("filterable_label_count").fill_null(0)
    ])
    if content_price_rate_review.height > 0:
        latest_price_info = (
            content_price_rate_review
            .sort(["content_id_hashed", "update_date"])
            .group_by("content_id_hashed")
            .last()
            .with_columns([
                ((pl.col("original_price") - pl.col("selling_price")) / pl.col("original_price") * 100).alias("discount_percentage"),
                pl.when(pl.col("selling_price") <= 50).then(pl.lit("low"))
                .when(pl.col("selling_price") <= 200).then(pl.lit("medium"))
                .when(pl.col("selling_price") <= 500).then(pl.lit("high"))
                .otherwise(pl.lit("premium")).alias("price_category"),
                (pl.col("content_review_wth_media_count") / pl.col("content_review_count")).fill_null(0).alias("review_media_ratio"),
                pl.when(pl.col("content_rate_avg") >= 4.5).then(pl.lit("excellent"))
                .when(pl.col("content_rate_avg") >= 4.0).then(pl.lit("good"))
                .when(pl.col("content_rate_avg") >= 3.0).then(pl.lit("average"))
                .otherwise(pl.lit("poor")).alias("rating_category")
            ])
        )
        content_features = content_features.join(latest_price_info.drop("update_date"), on="content_id_hashed", how="left")
    if content_search_filtered.height > 0:
        content_search_agg = (
            content_search_filtered
            .group_by("content_id_hashed")
            .agg([
                pl.col("total_search_impression").sum().alias("content_search_impressions"),
                pl.col("total_search_click").sum().alias("content_search_clicks"),
                (pl.col("total_search_click").sum() / pl.col("total_search_impression").sum()).alias("content_search_ctr"),
                pl.col("date").count().alias("content_search_days")
            ])
        )
        content_features = content_features.join(content_search_agg, on="content_id_hashed", how="left")
    if content_sitewide_filtered.height > 0:
        content_sitewide_agg = (
            content_sitewide_filtered
            .group_by("content_id_hashed")
            .agg([
                pl.col("total_click").sum().alias("content_total_clicks"),
                pl.col("total_cart").sum().alias("content_total_cart"),
                pl.col("total_fav").sum().alias("content_total_fav"),
                pl.col("total_order").sum().alias("content_total_orders"),
                (pl.col("total_cart").sum() / pl.col("total_click").sum()).alias("content_cart_rate"),
                (pl.col("total_order").sum() / pl.col("total_click").sum()).alias("content_order_rate"),
                pl.col("date").count().alias("content_active_days")
            ])
        )
        content_features = content_features.join(content_sitewide_agg, on="content_id_hashed", how="left")
    numeric_cols = [col for col in content_features.columns 
                   if col not in ["content_id_hashed", "level1_category_clean", "level2_category_clean", 
                                "leaf_category_clean", "cv_tags", "price_category", "rating_category"] 
                   and content_features[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
    content_features = content_features.with_columns([
        pl.col(col).fill_null(0) for col in numeric_cols
    ])
    return content_features

def create_interaction_features(session_df, user_features, content_features):
    enriched_data = (
        session_df
        .join(user_features, on="user_id_hashed", how="left")
        .join(content_features, on="content_id_hashed", how="left")
    )
    enriched_data = enriched_data.with_columns([
        pl.col("ts_hour").dt.hour().alias("hour_of_day"),
        pl.col("ts_hour").dt.weekday().alias("day_of_week"),
        pl.col("ts_hour").dt.day().alias("day_of_month"),
        (pl.col("ts_hour").dt.weekday() >= 5).alias("is_weekend"),
        pl.when(pl.col("ts_hour").dt.hour().is_between(6, 12)).then(pl.lit("morning"))
        .when(pl.col("ts_hour").dt.hour().is_between(12, 18)).then(pl.lit("afternoon"))
        .when(pl.col("ts_hour").dt.hour().is_between(18, 23)).then(pl.lit("evening"))
        .otherwise(pl.lit("night")).alias("time_period")
    ])
    return enriched_data

# --- 2. MODEL YÜKLEME VE SIRALAMA ---
def load_trained_models(click_model_path, order_model_path):
    model_click = lgb.Booster(model_file=click_model_path)
    model_order = lgb.Booster(model_file=order_model_path)
    return model_click, model_order

def prepare_features_for_model(enriched_df):
    # Sadece numeric ve hashlenmiş kategorik sütunlar
    numeric_feature_cols = [
        'user_age', 'user_tenure_in_days', 'user_total_clicks', 'user_total_cart', 'user_total_fav', 'user_total_orders', 
        'user_avg_clicks', 'user_activity_days', 'user_search_impressions', 'user_search_clicks', 'user_search_days', 'user_search_ctr',
        'attribute_type_count', 'total_attribute_option_count', 'merchant_count', 'filterable_label_count', 'content_age_days',
        'original_price', 'selling_price', 'discounted_price', 'content_review_count', 'content_rate_count', 'content_rate_avg',
        'discount_percentage', 'review_media_ratio', 'content_search_impressions', 'content_search_clicks', 'content_search_ctr',
        'content_search_days', 'content_total_clicks', 'content_total_cart', 'content_total_fav', 'content_total_orders', 'content_cart_rate', 'content_order_rate', 'content_active_days',
        'hour_of_day', 'day_of_week', 'day_of_month', 'search_term_length', 'search_term_word_count'
    ]
    boolean_cols = ['has_cv_tags', 'is_weekend']
    categorical_cols = ['user_gender_clean', 'user_tenure_category', 'level1_category_clean', 'level2_category_clean', 'leaf_category_clean', 'price_category', 'rating_category', 'time_period']
    all_features = numeric_feature_cols + boolean_cols + categorical_cols
    available_features = [col for col in all_features if col in enriched_df.columns]
    df_work = enriched_df.select(available_features + ['session_id', 'content_id_hashed']).clone()
    for col in categorical_cols:
        if col in df_work.columns:
            df_work = df_work.with_columns([
                (pl.col(col).cast(pl.String).fill_null("MISSING").hash(seed=42) % 1000000).cast(pl.Int32).alias(col)
            ])
    for col in boolean_cols:
        if col in df_work.columns:
            df_work = df_work.with_columns([
                pl.col(col).cast(pl.Int32).alias(col)
            ])
    for col in numeric_feature_cols:
        if col in df_work.columns:
            df_work = df_work.with_columns([
                pl.col(col).cast(pl.Float32).fill_null(0.0).alias(col)
            ])
    df_pandas = df_work.to_pandas()
    X = df_pandas[available_features].copy()
    X = X.fillna(0)
    return X, df_pandas['session_id'].values, df_pandas['content_id_hashed'].values

def predict_and_rank(model_click, model_order, X, session_ids, content_ids, output_path):
    click_pred = model_click.predict(X, num_iteration=model_click.best_iteration)
    order_pred = model_order.predict(X, num_iteration=model_order.best_iteration)
    combined_pred = 0.3 * click_pred + 0.7 * order_pred
    df = pd.DataFrame({
        'session_id': session_ids,
        'content_id': content_ids,
        'score': combined_pred
    })
    # Her session için content'leri skora göre sırala ve boşlukla birleştir
    submission = (
        df.sort_values(['session_id', 'score'], ascending=[True, False])
        .groupby('session_id')['content_id']
        .apply(lambda x: ' '.join(x))
        .reset_index()
    )
    submission.columns = ['session_id', 'prediction']
    submission.to_csv(output_path, index=False)
    print(f"Submission dosyası '{output_path}' olarak kaydedildi. Toplam session: {submission.shape[0]}")
    return submission

# --- 3. ANA ÇALIŞTIRMA ---
if __name__ == "__main__":
    DATA_PATH = "trendyol-e-ticaret-hackathonu-2025-kaggle/data"
    CLICK_MODEL_PATH = "model_click_comprehensive.txt"
    ORDER_MODEL_PATH = "model_order_comprehensive.txt"
    OUTPUT_PATH = "submission_inference.csv"

    print("Veriler yükleniyor...")
    data = load_data(DATA_PATH)
    print("Test feature engineering başlıyor...")
    user_features = create_user_features(data['user_metadata'], data['user_sitewide_log'], data['user_search_log'], data['test_sessions'])
    content_features = create_content_features(data['content_metadata'], data['content_price_rate_review'], data['content_search_log'], data['content_sitewide_log'], data['test_sessions'])
    enriched = create_interaction_features(data['test_sessions'], user_features, content_features)
    print("Model input hazırlanıyor...")
    X, session_ids, content_ids = prepare_features_for_model(enriched)
    print("Modeller yükleniyor...")
    model_click, model_order = load_trained_models(CLICK_MODEL_PATH, ORDER_MODEL_PATH)
    print("Tahmin ve sıralama başlıyor...")
    predict_and_rank(model_click, model_order, X, session_ids, content_ids, OUTPUT_PATH)
    print("TAMAMLANDI!")
