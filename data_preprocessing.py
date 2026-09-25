"""
The following preprocessing operations are performed on the datasets in the CLIMB database.
For the specific operations and the preprocessing of other datasets, see the appendix of the main text.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml


def load_and_preprocess_dataset(dataset_name_or_id, is_climb: bool = True, local_csv_path: str = None) -> pd.DataFrame:
    """
    Unified ingestion pipeline for benchmark tabular datasets.
    """
    if local_csv_path is not None:
        data = pd.read_csv(local_csv_path)
        X = data.iloc[:, :-1].copy()
        y = data.iloc[:, -1].copy()
    elif is_climb:
        dataset = fetch_openml(
            name=dataset_name_or_id if not str(dataset_name_or_id).isdigit() else None,
            data_id=int(dataset_name_or_id) if str(dataset_name_or_id).isdigit() else None,
            as_frame=True,
            parser="auto",
        )
        X = dataset.data.copy()
        y = dataset.target.copy()
    else:
        raise ValueError("Invalid data source configuration.")

    # 1. String Sanitization: standardize dirty tokens to NaN
    X = X.replace(r"^\s*\?\s*$", np.nan, regex=True).replace("?", np.nan)
    y = y.replace(r"^\s*\?\s*$", np.nan, regex=True).replace("?", np.nan)

    # 2. Type Disambiguation: resolve object columns corrupted by missing tokens
    for col in X.columns:
        s_numeric = pd.to_numeric(X[col], errors="coerce")
        orig_valid_count = X[col].dropna().shape[0]
        if orig_valid_count > 0 and s_numeric.dropna().shape[0] == orig_valid_count:
            X[col] = s_numeric
        else:
            X[col] = X[col].astype(str)

    # 3. Stratified Missing Value Imputation
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            median_val = X[col].median()
            fill_val = 0.0 if pd.isna(median_val) else median_val
            X[col] = X[col].fillna(fill_val)
        else:
            mode_series = X[col].dropna().mode()
            fill_val = mode_series[0] if len(mode_series) > 0 else "missing"
            X[col] = X[col].fillna(fill_val).astype(str)

    # 4. Label Binarization: minority mapped to 1, majority mapped to 0
    valid_idx = y.dropna().index
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]

    label_counts = y.value_counts()
    minority_class_raw = label_counts.idxmin()
    y_standardized = pd.Series(
        np.where(y == minority_class_raw, 1, 0), index=y.index, name="target"
    )

    # 5. Full One-Hot Encoding: drop_first=False preserves complete group coordinates
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if len(cat_cols) > 0:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=False, dtype=float)

    # Final cast to ensure complete floating-point design matrix
    X = X.fillna(0.0).astype(float)
    return pd.concat([X, y_standardized], axis=1)