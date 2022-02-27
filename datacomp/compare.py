import sys
from pathlib import Path

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class ColumnResult:
    dtype_match: bool = True
    left_dtype: Any = None
    right_dtype: Any = None
    mismatch_percent: Optional[float] = None
    mismatch_number: Optional[int] = None
    mismatch_data: Optional[pd.DataFrame] = None


@dataclass
class CompareResult:
    match: bool = True
    data_match: bool = True
    left_only_columns: Optional[List[str]] = None
    right_only_columns: Optional[List[str]] = None
    index_match: bool = True
    left_index_duplicates: Optional[pd.DataFrame] = None
    right_index_duplicates: Optional[pd.DataFrame] = None
    common_index_count: int = 0
    left_index_count: int = 0
    right_index_count: int = 0
    left_only_indexes: Optional[pd.Series] = None
    right_only_indexes: Optional[pd.Series] = None
    column_results: Optional[Dict[str, ColumnResult]] = None

    def __post_init__(self):
        self.column_results = {}
        self.left_only_indexes = pd.Series([], dtype=str)
        self.right_only_indexes = pd.Series([], dtype=str)


def series_nonequal_index(ser1: pd.Series, ser2: pd.Series) -> pd.Series:
    ser1_null = ser1.isnull()
    ser2_null = ser2.isnull()
    # Get index where series values are null in one series but the other
    nonequal_null_idx = list(set(ser1[ser1_null].index) ^ set(ser2[ser2_null].index))
    # Get index where series non-null values are nonequal
    notnull_idx = list(set(ser1[~ser1_null].index) & set(ser2[~ser2_null].index))
    nonequal_notnull_idx = ser1.loc[notnull_idx] != ser2.loc[notnull_idx]
    nonequal_notnull_idx = ser1.loc[notnull_idx][nonequal_notnull_idx].index.tolist()
    # Combine null/non-null indexes
    return nonequal_null_idx + nonequal_notnull_idx


def load_file(fpath: Path) -> pd.DataFrame:
    if fpath.suffix == ".parquet":
        return pd.read_parquet(fpath)
    if fpath.suffix == ".csv":
        return pd.read_csv(fpath)

    raise RuntimeError(f"Unsupported file type {fpath.suffix} for {fpath}")


def compare_data(data_fpath_1: Path, data_fpath_2: Path, index_cols: List[str]) -> CompareResult:
    result = CompareResult()

    print(f"Loading {data_fpath_1} ...")
    df1 = load_file(data_fpath_1)
    print(f"Loading {data_fpath_2} ...")
    df2 = load_file(data_fpath_2)

    if set(df1.columns) != set(df2.columns):
        result.match = False
        result.left_only_columns = sorted(set(df1.columns) - set(df2.columns))
        result.right_only_columns = sorted(set(df2.columns) - set(df1.columns))

    if index_cols:
        print("Indexing dataframes ...")
        for df, data_fpath in [(df1, data_fpath_1), (df2, data_fpath_2)]:
            try:
                df.sort_values(index_cols, inplace=True)
                df.set_index(index_cols, inplace=True)
            except KeyError:
                print()
                print(
                    f"⚠️  ERROR: index columns ({', '.join(index_cols)}) not found in {data_fpath}"
                )
                sys.exit(1)

        print("Validating dataframe indices ...")
        l_dup_indices = df1.index.duplicated(keep=False)
        r_dup_indices = df2.index.duplicated(keep=False)
        if l_dup_indices.any():
            # Consider this an error because we are unable to determine if the data matches
            result.match = False
            result.left_index_duplicates = pd.DataFrame(
                df.index[l_dup_indices].value_counts().rename("count")
            )
            df1 = df1[~df1.index.duplicated(keep="first")]
        if r_dup_indices.any():
            result.match = False
            result.right_index_duplicates = pd.DataFrame(
                df.index[r_dup_indices].value_counts().rename("count")
            )
            df2 = df2[~df2.index.duplicated(keep="first")]

    print("Comparing dataframe indices ...")
    result.left_index_count = len(df1)
    result.right_index_count = len(df2)
    if not df1.index.equals(df2.index):
        result.match = False
        result.index_match = False

        result.left_only_indexes = pd.Series(
            sorted(set(df1.index) - set(df2.index)), dtype=df1.index.dtype
        )
        result.right_only_indexes = pd.Series(
            sorted(set(df2.index) - set(df1.index)), dtype=df2.index.dtype
        )

        common_ids = list(set(df1.index) & set(df2.index))
        result.common_index_count = len(common_ids)

        if not common_ids:
            return result

        df1 = df1.loc[common_ids]
        df2 = df2.loc[common_ids]
    else:
        result.common_index_count = result.left_index_count

    common_cols = [col for col in df1 if col in set(df2.columns)]
    # quick check if dataframes match, otherwise show detailed differences
    print("Comparing data ...")
    if not df1[common_cols].equals(df2[common_cols]):
        result.match = False
        result.data_match = False

        for idx, col in enumerate(common_cols):
            print(f"Comparing column ({idx + 1}/{len(common_cols)}): {col} ...")
            if not df1[col].equals(df2[col]):

                mismatch_idx = series_nonequal_index(df1[col], df2[col])
                mismatch_df = df1.loc[mismatch_idx, [col]].join(
                    df2.loc[mismatch_idx, [col]], lsuffix="_LEFT", rsuffix="_RIGHT"
                )
                mismatch_pct = len(mismatch_idx) / len(df1) * 100

                result.column_results[col] = ColumnResult(
                    dtype_match=(df1[col].dtype == df2[col].dtype),
                    left_dtype=df1[col].dtype,
                    right_dtype=df2[col].dtype,
                    mismatch_percent=mismatch_pct,
                    mismatch_number=len(mismatch_idx),
                    mismatch_data=mismatch_df,
                )

    return result
