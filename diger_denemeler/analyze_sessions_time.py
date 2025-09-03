import sys
from pathlib import Path
import pandas as pd

# Inputs: paths to train/test parquet. Output: print schema and time distributions.

paths = [
    Path(r"c:\\Users\\pc\\Desktop\\trendyol_hekaton\\YeniDeneme\\veri\\train_sessions.parquet"),
    Path(r"c:\\Users\\pc\\Desktop\\trendyol_hekaton\\trendyol-e-ticaret-hackathonu-2025-kaggle\\data\\train_sessions.parquet"),
    Path(r"c:\\Users\\pc\\Desktop\\trendyol_hekaton\\YeniDeneme\\veri\\test_sessions.parquet"),
    Path(r"c:\\Users\\pc\\Desktop\\trendyol_hekaton\\trendyol-e-ticaret-hackathonu-2025-kaggle\\data\\test_sessions.parquet"),
]

# Deduplicate by existing
paths = [p for p in paths if p.exists()]
if not paths:
    print("No parquet files found.")
    sys.exit(0)

# Heuristics for time columns
TIME_CANDIDATES = {"event_time", "timestamp", "ts", "time", "datetime", "event_timestamp", "session_start", "session_time"}

for p in paths:
    print(f"\n=== File: {p} ===")
    df = pd.read_parquet(p)
    print("Rows:", len(df), "Cols:", len(df.columns))
    print("Columns:", list(df.columns))

    # Find time-like columns
    time_cols = []
    for c in df.columns:
        lc = str(c).lower()
        if lc in TIME_CANDIDATES or any(tok in lc for tok in ["time", "date", "ts"]):
            time_cols.append(c)
    # Try to parse
    parsed = None
    time_col = None
    for c in time_cols:
        try:
            s = pd.to_datetime(df[c], errors='coerce', utc=True)
            if s.notna().sum() > 0:
                parsed = s
                time_col = c
                break
        except Exception:
            pass
    if parsed is None:
        # try the first numeric column as epoch
        for c in df.select_dtypes(include=['int64','int32','float64','float32']).columns:
            s = pd.to_datetime(df[c], unit='s', errors='coerce', utc=True)
            if s.notna().sum() > 0:
                parsed = s
                time_col = c
                break
    if parsed is None:
        print("No timestamp-like column detected.")
        continue

    dt = parsed.dt.tz_convert("UTC") if parsed.dt.tz is not None else parsed.dt.tz_localize("UTC")
    print(f"Detected time column: {time_col}")
    print("Time span UTC:")
    print("  min:", dt.min())
    print("  max:", dt.max())

    # Daily counts
    by_day = dt.dt.floor('D').value_counts().sort_index()
    print("\nDaily counts (UTC) - top 10:")
    print(by_day.head(10))
    print("... last 10:")
    print(by_day.tail(10))

    # Hourly counts
    by_hour = dt.dt.floor('H').value_counts().sort_index()
    print("\nHourly counts (UTC) - first 24:")
    print(by_hour.head(24))
    print("... last 24:")
    print(by_hour.tail(24))

    # Distribution by hour of day and day of week
    hod = dt.dt.hour.value_counts().sort_index()
    dow = dt.dt.dayofweek.value_counts().sort_index()
    print("\nCount by hour-of-day (0-23):")
    print(hod)
    print("\nCount by day-of-week (Mon=0):")
    print(dow)
