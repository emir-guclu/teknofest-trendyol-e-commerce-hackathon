"""
Script to preprocess session data for DeepFM model.
Loads processed_train_sessions.parquet and processed_test_sessions.parquet,
encodes categorical features, scales numeric ones, and saves:
  - features and labels as NumPy arrays
  - encoders and scalers
  - a meta_deepfm.json describing feature fields and sizes
    alt_train = TOOLS_VERI / "train_sessions.parquet"
    alt_test = TOOLS_VERI / "test_sessions.parquet"
    if alt_train.exists() and alt_test.exists():
        print("Using precomputed vector sessions from tools/veri")
        TRAIN_PROC = alt_train
        TEST_PROC = alt_test
        PRECOMPUTED = True
    else:
        TRAIN_PROC = DATA_DIR / "processed_train_sessions.parquet"
        TEST_PROC = DATA_DIR / "processed_test_sessions.parquet"
        PRECOMPUTED = False

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "trendyol-e-ticaret-hackathonu-2025-kaggle" / "data"
# Fallback to precomputed vector sessions in tools/veri
TOOLS_VERI = ROOT / "tools" / "veri"
PRECOMPUTED = False
if TOOLS_VERI.exists():
    alt_train = TOOLS_VERI / "train_sessions.parquet"
    alt_test = TOOLS_VERI / "test_sessions.parquet"
        if alt_train.exists() and alt_test.exists():
            print("Using precomputed vector sessions from tools/veri")
            TRAIN_PROC = alt_train
            TEST_PROC = alt_test
            PRECOMPUTED = True
    else:
        TRAIN_PROC = DATA_DIR / "processed_train_sessions.parquet"
        TEST_PROC = DATA_DIR / "processed_test_sessions.parquet"
else:
    TRAIN_PROC = DATA_DIR / "processed_train_sessions.parquet"
    TEST_PROC = DATA_DIR / "processed_test_sessions.parquet"
OUT_DIR = DATA_DIR / "deepfm"
OUT_DIR.mkdir(exist_ok=True, parents=True)

# Output files
ENCODERS_FILE = OUT_DIR / "cat_encoders.pkl"
SCALER_FILE = OUT_DIR / "num_scaler.pkl"
META_FILE = OUT_DIR / "meta_deepfm.json"
X_TRAIN_FILE = OUT_DIR / "X_train.npy"
X_TEST_FILE = OUT_DIR / "X_test.npy"
Y_TRAIN_FILE = OUT_DIR / "y_train.npy"
Y_ORDER_FILE = OUT_DIR / "y_order.npy"


def main():
    # Load processed data or skip if precomputed
    if PRECOMPUTED:
        print("Skipping preprocessing, using precomputed sessions")
        train = pd.read_parquet(TRAIN_PROC)
        test = pd.read_parquet(TEST_PROC)
        # labels
        y_click = train['clicked'].values.astype(int)
        y_order = train['ordered'].values.astype(int)
        # drop id and labels
        X_train = train.drop(columns=['session_id','user_id_hashed','content_id_hashed','clicked','ordered'], errors='ignore')
        X_test = test.drop(columns=['session_id','user_id_hashed','content_id_hashed'], errors='ignore')
        # save numpy arrays directly
        X_train_all = X_train.values.astype(np.float32)
        X_test_all = X_test.values.astype(np.float32)
        np.save(X_TRAIN_FILE, X_train_all)
        np.save(X_TEST_FILE, X_test_all)
        np.save(Y_TRAIN_FILE, y_click)
        np.save(Y_ORDER_FILE, y_order)
        # save empty encoders and scaler meta
        with open(ENCODERS_FILE, 'wb') as f:
            pickle.dump({'encoders': {}, 'cat_cols': []}, f)
        with open(SCALER_FILE, 'wb') as f:
            pickle.dump({'scaler': None, 'num_cols': X_train.columns.tolist()}, f)
        meta = {
            'cat_cols': [],
            'num_cols': X_train.columns.tolist(),
            'feature_sizes': [1] * len(X_train.columns),
            'field_size': len(X_train.columns),
            'task': ['click', 'order']
        }
        with open(META_FILE, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print("Precomputed deepfm data saved. Outputs in:", OUT_DIR)
        return
    train = pd.read_parquet(TRAIN_PROC)
    test = pd.read_parquet(TEST_PROC)

    # Labels
    y_click = train['clicked'].values.astype(int)
    y_order = train['ordered'].values.astype(int)

    # Drop identifiers and labels
    drop_cols = ['session_id', 'user_id_hashed', 'content_id_hashed', 'clicked', 'added_to_cart', 'added_to_fav', 'ordered']
    # Drop identifier and label columns, ignore missing ones
    X_train = train.drop(columns=drop_cols, errors='ignore')
    X_test = test.drop(columns=['session_id', 'user_id_hashed', 'content_id_hashed'], errors='ignore')

    # Missing value strategy: choose 'statistic', 'flag', or 'none' via sys.argv
    import sys
    if len(sys.argv) > 1:
        missing_strategy = sys.argv[1]
    else:
        missing_strategy = 'none'
    print(f"Missing value strategy: {missing_strategy}")
    # Identify initial numeric and categorical columns
    num_init_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_init_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    # Apply imputation or flags
    if missing_strategy == 'statistic':
        # Numeric: fill with median
        for col in num_init_cols:
            med = X_train[col].median()
            X_train[col] = X_train[col].fillna(med)
            X_test[col] = X_test[col].fillna(med)
        # Categorical: fill with mode
        for col in cat_init_cols:
            mode_val = X_train[col].mode().iloc[0] if not X_train[col].mode().empty else 'unknown'
            X_train[col] = X_train[col].fillna(mode_val)
            X_test[col] = X_test[col].fillna(mode_val)
    elif missing_strategy == 'flag':
        # Numeric: create missing flags and fill zeros
        for col in num_init_cols:
            flag = f"{col}_missing"
            X_train[flag] = X_train[col].isnull().astype(int)
            X_test[flag] = X_test[col].isnull().astype(int)
            X_train[col] = X_train[col].fillna(0)
            X_test[col] = X_test[col].fillna(0)
        # Categorical: create missing flags and label missing
        for col in cat_init_cols:
            flag = f"{col}_missing"
            X_train[flag] = X_train[col].isnull().astype(int)
            X_test[flag] = X_test[col].isnull().astype(int)
            X_train[col] = X_train[col].fillna('missing').astype(str)
            X_test[col] = X_test[col].fillna('missing').astype(str)
    # else: default handling later

    # Identify and encode multi-word text features via transformer embeddings
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise SystemExit(
            "Please install sentence-transformers to encode text features: `pip install sentence-transformers`"
        )
    # Detect text columns with whitespace (multi-word descriptions)
    text_cols = [col for col in X_train.select_dtypes(include=['object']).columns
                 if X_train[col].fillna('').str.contains(r'\s+').any()]
    emb_features: List[str] = []
    st_model = SentenceTransformer('all-MiniLM-L6-v2')
    for col in text_cols:
        # Prepare texts
        train_texts = X_train[col].fillna('unknown').astype(str).tolist()
        test_texts = X_test.get(col, pd.Series('', index=X_test.index)).fillna('unknown').astype(str).tolist()
        # Compute embeddings
        emb_train = st_model.encode(train_texts, show_progress_bar=False)
        emb_test = st_model.encode(test_texts, show_progress_bar=False)
        # Append embedding dimensions as numeric columns
        dim = emb_train.shape[1]
        for i in range(dim):
            fname = f"{col}_emb_{i}"
            X_train[fname] = emb_train[:, i]
            X_test[fname] = emb_test[:, i]
            emb_features.append(fname)
    # Drop original text columns from categorical
    X_train = X_train.drop(columns=text_cols)
    X_test = X_test.drop(columns=text_cols)
    # Define categorical and numeric columns
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    # Numeric cols include transformer embeddings and existing numeric features
    num_cols = emb_features + X_train.select_dtypes(include=[np.number]).columns.drop(emb_features).tolist()

    print(f"Categorical columns ({len(cat_cols)}): {cat_cols}")
    print(f"Numeric columns ({len(num_cols)}): {num_cols}")

    # Encode categorical features
    encoders = {}
    X_train_cat = np.zeros((len(X_train), len(cat_cols)), dtype=int)
    X_test_cat = np.zeros((len(X_test), len(cat_cols)), dtype=int)
    for i, col in enumerate(cat_cols):
        # Combine for fitting
        combined = pd.concat([X_train[col], X_test.get(col, pd.Series([], dtype=str))], axis=0).fillna('unknown').astype(str)
        le = LabelEncoder().fit(combined)
        encoders[col] = le
        X_train_cat[:, i] = le.transform(X_train[col].fillna('unknown').astype(str))
        X_test_cat[:, i] = le.transform(X_test[col].fillna('unknown').astype(str))

    # Scale numeric features
    scaler = MinMaxScaler()
    X_train_num = scaler.fit_transform(X_train[num_cols].fillna(0))
    X_test_num = scaler.transform(X_test[num_cols].fillna(0))

    # Combine features
    X_train_all = np.hstack([X_train_cat, X_train_num])
    X_test_all = np.hstack([X_test_cat, X_test_num])

    # Save arrays
    np.save(X_TRAIN_FILE, X_train_all)
    np.save(X_TEST_FILE, X_test_all)
    np.save(Y_TRAIN_FILE, y_click)
    np.save(Y_ORDER_FILE, y_order)

    # Save encoders and scaler
    with open(ENCODERS_FILE, 'wb') as f:
        pickle.dump({'encoders': encoders, 'cat_cols': cat_cols}, f)
    with open(SCALER_FILE, 'wb') as f:
        pickle.dump({'scaler': scaler, 'num_cols': num_cols}, f)

    # Write meta information: for DeepFM, feature fields and their sizes
    feature_sizes = [len(encoders[col].classes_) for col in cat_cols] + [1] * len(num_cols)
    meta = {
        'cat_cols': cat_cols,
        'num_cols': num_cols,
        'feature_sizes': feature_sizes,
        'field_size': len(cat_cols) + len(num_cols),
        'task': ['click', 'order']
    }
    with open(META_FILE, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("DeepFM data preprocessing complete. Outputs saved in:")
    print(OUT_DIR)


if __name__ == '__main__':
    main()
