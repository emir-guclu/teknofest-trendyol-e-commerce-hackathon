import os
import re
import math
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pyarrow.dataset as ds
except Exception as e:
    raise SystemExit(
        "PyArrow is required for this script. Please install pyarrow. Error: %s" % e
    )


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "trendyol-e-ticaret-hackathonu-2025-kaggle" / "data"
REPORT_MD = ROOT / "trendyol-e-ticaret-hackathonu-2025-kaggle" / "data_schema_report.md"
REPORT_CSV = ROOT / "trendyol-e-ticaret-hackathonu-2025-kaggle" / "data_schema_report.csv"


def list_parquet_files() -> List[Path]:
    # Known 14 parquet files as per repo structure
    expected = [
        DATA_ROOT / "train_sessions.parquet",
        DATA_ROOT / "test_sessions.parquet",
        DATA_ROOT / "term" / "search_log.parquet",
        DATA_ROOT / "content" / "search_log.parquet",
        DATA_ROOT / "content" / "sitewide_log.parquet",
        DATA_ROOT / "content" / "price_rate_review_data.parquet",
        DATA_ROOT / "content" / "metadata.parquet",
        DATA_ROOT / "content" / "top_terms_log.parquet",
        DATA_ROOT / "user" / "search_log.parquet",
        DATA_ROOT / "user" / "sitewide_log.parquet",
        DATA_ROOT / "user" / "fashion_search_log.parquet",
        DATA_ROOT / "user" / "fashion_sitewide_log.parquet",
        DATA_ROOT / "user" / "metadata.parquet",
        DATA_ROOT / "user" / "top_terms_log.parquet",
    ]
    return [p for p in expected if p.exists()]


ID_LIKE_RE = re.compile(r"(^|_)(id|sku|user|item|product|category|merchant|brand)(s)?(_id)?($|_)", re.IGNORECASE)


def infer_role(col: str, series: pd.Series, arrow_type: pa.DataType) -> Tuple[str, str]:
    """Return (role, note) where role in {numeric, categorical}.

    Heuristics:
    - bool -> categorical
    - datetime/timestamp -> numeric (feature engineered), note = 'datetime'
    - strings -> categorical
    - numeric:
        - id-like names -> categorical
        - few unique values (<=20) -> categorical
        - low unique ratio (<=5%) and n_unique<=100 -> categorical
        - else numeric
    """

    note = ""
    # Normalize pandas dtype info
    pd_dtype = series.dtype
    is_bool = pd.api.types.is_bool_dtype(pd_dtype)
    is_datetime = pd.api.types.is_datetime64_any_dtype(pd_dtype) or pa.types.is_timestamp(arrow_type)
    is_numeric = pd.api.types.is_numeric_dtype(pd_dtype) and not is_bool
    is_string_like = pd.api.types.is_string_dtype(pd_dtype) or pa.types.is_string(arrow_type)

    n = len(series)
    # Use non-null values for uniqueness to avoid NaN counting
    non_null = series.dropna()
    try:
        n_unique = int(non_null.nunique(dropna=True))
    except Exception:
        # Some complex types may not be hashable
        n_unique = math.nan
    unique_ratio = (n_unique / len(non_null)) if len(non_null) > 0 and isinstance(n_unique, (int, float)) else math.nan

    if is_bool:
        return "categorical", "bool"

    if is_datetime:
        return "numeric", "datetime"

    if is_string_like:
        return "categorical", "string"

    if is_numeric:
        if ID_LIKE_RE.search(col):
            return "categorical", "id-like"

        if isinstance(n_unique, (int, float)):
            if n_unique <= 20:
                return "categorical", f"low-cardinality ({n_unique})"
            if not math.isnan(unique_ratio) and unique_ratio <= 0.05 and n_unique <= 100:
                return "categorical", f"low unique ratio ({unique_ratio:.3f})"
        return "numeric", "continuous"

    # Fallback
    return "categorical", f"arrow={arrow_type}, pandas={pd_dtype}"


def sample_table(path: Path, columns: List[str]) -> pd.DataFrame:
    """Read a sample of up to max_rows rows for the given columns using pyarrow dataset for efficiency."""
    max_rows = 100_000
    try:
        dataset = ds.dataset(str(path))
        scanner = dataset.scanner(columns=columns)
        tbl = scanner.head(max_rows)
        return tbl.to_pandas(types_mapper=pd.ArrowDtype)
    except Exception:
        # Fallback to pandas
        return pd.read_parquet(path, columns=columns).head(max_rows)


def analyze_file(path: Path) -> List[Dict[str, str]]:
    pf = pq.ParquetFile(path)
    schema = pf.schema_arrow
    cols = [f.name for f in schema]

    # Sample data frame for statistics
    df = sample_table(path, cols)

    rows: List[Dict[str, str]] = []
    for field in schema:
        col = field.name
        arrow_type = field.type
        series = df[col] if col in df.columns else pd.Series([], dtype="float64")
        role, note = infer_role(col, series, arrow_type)
        missing = int(series.isna().sum()) if col in df.columns else 0
        n = len(series)
        try:
            n_unique = int(series.nunique(dropna=True))
        except Exception:
            n_unique = 0
        rows.append(
            {
                "file": str(path.relative_to(DATA_ROOT)),
                "column": col,
                "arrow_type": str(arrow_type),
                "pandas_dtype": str(series.dtype) if col in df.columns else "",
                "non_null_rows": str(n - missing),
                "unique_values_in_sample": str(n_unique),
                "role": role,
                "note": note,
            }
        )
    return rows


def write_markdown(all_rows: List[Dict[str, str]]):
    by_file: Dict[str, List[Dict[str, str]]] = {}
    for r in all_rows:
        by_file.setdefault(r["file"], []).append(r)

    lines: List[str] = []
    lines.append("# Veri Şeması ve Modelleme İçin Sütun Rolleri")
    lines.append("")
    lines.append("Bu rapor 14 parquet dosyasındaki sütunların veri tiplerini ve eğitimde 'sayısal' ya da 'kategorik' olarak önerilen rollerini özetler.")
    lines.append("")
    for f in sorted(by_file.keys()):
        lines.append(f"## {f}")
        lines.append("")
        lines.append("| Sütun | Arrow Tipi | Pandas Dtype | Örnek Non-Null | Örnek Unique | Rol | Not |")
        lines.append("|---|---|---|---:|---:|---|---|")
        for r in sorted(by_file[f], key=lambda x: x["column"].lower()):
            lines.append(
                "| {column} | {arrow_type} | {pandas_dtype} | {non_null_rows} | {unique_values_in_sample} | {role} | {note} |".format(
                    **r
                )
            )
        lines.append("")

    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def write_csv(all_rows: List[Dict[str, str]]):
    df = pd.DataFrame(all_rows)
    df.to_csv(REPORT_CSV, index=False, encoding="utf-8")


def main():
    files = list_parquet_files()
    if not files:
        raise SystemExit(f"No parquet files found under {DATA_ROOT}")

    all_rows: List[Dict[str, str]] = []
    for p in files:
        try:
            rows = analyze_file(p)
            all_rows.extend(rows)
            print(f"Analyzed: {p}")
        except Exception as e:
            print(f"ERROR analyzing {p}: {e}")

    write_markdown(all_rows)
    write_csv(all_rows)
    print(f"\nReport written to:\n - {REPORT_MD}\n - {REPORT_CSV}")


if __name__ == "__main__":
    main()
