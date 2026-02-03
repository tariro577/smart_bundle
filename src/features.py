from __future__ import annotations

import pandas as pd

RAW_NUMERIC_COLS = [
    "Account length",
    "Total day minutes",
    "Total day calls",
    "Total eve minutes",
    "Total eve calls",
    "Total night minutes",
    "Total night calls",
    "Total intl minutes",
    "Total intl calls",
    "Number vmail messages",
    "Customer service calls",
]

RAW_BINARY_COLS = [
    "International plan",
    "Voice mail plan",
]

TARGET_COL = "Bundle"


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [col.strip() for col in df.columns]
    # Normalize common lowercase Kaggle column names to expected format
    normalized = {col.lower(): col for col in df.columns}
    rename_map = {}
    for col in RAW_NUMERIC_COLS + RAW_BINARY_COLS:
        lower = col.lower()
        if lower in normalized and normalized[lower] != col:
            rename_map[normalized[lower]] = col
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_column_names(df)

    # Drop duplicate rows to avoid skewing patterns
    df = df.drop_duplicates().reset_index(drop=True)

    # Standardize boolean-like columns
    for col in RAW_BINARY_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().map({"yes": 1, "no": 0})

    # Coerce numeric columns
    for col in RAW_NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill missing numeric values with median per column
    numeric_cols = [col for col in RAW_NUMERIC_COLS if col in df.columns]
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Fill missing binary columns with mode
    for col in RAW_BINARY_COLS:
        if col in df.columns:
            mode = df[col].mode(dropna=True)
            df[col] = df[col].fillna(mode.iloc[0] if not mode.empty else 0)

    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Total minutes"] = (
        df["Total day minutes"]
        + df["Total eve minutes"]
        + df["Total night minutes"]
    )
    df["Total calls"] = (
        df["Total day calls"]
        + df["Total eve calls"]
        + df["Total night calls"]
    )
    df["Avg minutes per call"] = df["Total minutes"] / df["Total calls"].replace(0, 1)
    df["Intl usage ratio"] = df["Total intl minutes"] / df["Total minutes"].replace(0, 1)
    df["Support calls per month"] = df["Customer service calls"] / df["Account length"].replace(0, 1)

    # Proxy for data usage (dataset lacks direct data usage)
    df["Data proxy"] = df["Total eve minutes"] * 0.6 + df["Total night minutes"] * 0.4

    return df


def assign_bundle(row: pd.Series) -> str:
    total_minutes = row["Total minutes"]
    intl_minutes = row["Total intl minutes"]
    intl_ratio = row["Intl usage ratio"]
    data_proxy = row["Data proxy"]
    vmail = row["Voice mail plan"]

    if intl_minutes >= 10 or intl_ratio >= 0.08:
        return "International Saver"

    if total_minutes >= 600 or data_proxy >= 180:
        return "Unlimited Voice + Data"

    if total_minutes >= 420 or data_proxy >= 140:
        return "Talk & Surf Plus"

    if total_minutes >= 260:
        return "Balanced Connect"

    if vmail == 1:
        return "Voice Essentials"

    return "Starter Lite"


def build_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_data(df)
    df = add_features(df)

    df[TARGET_COL] = df.apply(assign_bundle, axis=1)
    return df


def model_features(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [
        "Account length",
        "Total day minutes",
        "Total day calls",
        "Total eve minutes",
        "Total eve calls",
        "Total night minutes",
        "Total night calls",
        "Total intl minutes",
        "Total intl calls",
        "Number vmail messages",
        "Customer service calls",
        "International plan",
        "Voice mail plan",
        "Total minutes",
        "Total calls",
        "Avg minutes per call",
        "Intl usage ratio",
        "Support calls per month",
        "Data proxy",
    ]
    return df[feature_cols]
