from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class ColumnResult:
    dtype_match: bool = True
    left_dtype: Any = None
    right_dtype: Any = None
    mismatch_percent: Optional[float] = None
    mismatch_number: Optional[int] = None
    mismatch_data: Optional[pd.DataFrame] = None
    diff_min: Optional[float] = None
    diff_max: Optional[float] = None
    diff_mean: Optional[float] = None
    diff_std: Optional[float] = None
    ratio_min: Optional[float] = None
    ratio_max: Optional[float] = None
    ratio_mean: Optional[float] = None
    ratio_std: Optional[float] = None
    ratio_min_nonzero: Optional[float] = None
    ratio_max_nonzero: Optional[float] = None
    ratio_inf_count: Optional[int] = None


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


def series_nonequal_index(
    ser1: pd.Series, ser2: pd.Series, atol: float = 0.0, rtol: float = 0.0
) -> pd.Series:
    ser1_null = ser1.isnull()
    ser2_null = ser2.isnull()
    # Get index where series values are null in one series but the other
    nonequal_null_idx = ser1[ser1_null ^ ser2_null].index.tolist()
    # Get index where series non-null values are nonequal
    notnull_idx = (~ser1_null) & (~ser2_null)

    if notnull_idx.any():
        if (
            isinstance(ser1.loc[ser1.notnull().index[0]], np.ndarray)
            or isinstance(ser2.loc[ser2.notnull().index[0]], np.ndarray)
        ):
            def compare_arrays(a1, a2):
                return np.allclose(a1, a2, atol=atol, rtol=rtol, equal_nan=True)

            # Use this function instead of != operator on ser1/ser2
            # (handle series of numpy arrays)
            v_array_equal = np.vectorize(compare_arrays)
            nonequal_notnull_idx = ~v_array_equal(ser1[notnull_idx], ser2[notnull_idx])
        elif (
            pd.api.types.is_numeric_dtype(ser1.dtype)
            and pd.api.types.is_numeric_dtype(ser2.dtype)
            and (atol > 0 or rtol > 0)
        ):
            # Otherwise if comparing numeric values and either "tolerance" option is set
            nonequal_notnull_idx = ~np.isclose(
                ser1[notnull_idx],
                ser2[notnull_idx],
                atol=atol,
                rtol=rtol,
            )
        else:
            # Otherwise, use much faster != operator
            nonequal_notnull_idx = ser1[notnull_idx] != ser2[notnull_idx]

        nonequal_notnull_idx = ser1[notnull_idx][nonequal_notnull_idx].index.tolist()
    else:
        nonequal_notnull_idx = []

    # Combine null/non-null indexes
    return nonequal_null_idx + nonequal_notnull_idx


def compare_data(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    index_cols: List[str],
    atol: float = 0,
    rtol: float = 0,
) -> CompareResult:
    result = CompareResult()

    if set(df1.columns) != set(df2.columns):
        result.match = False
        result.left_only_columns = sorted(set(df1.columns) - set(df2.columns))
        result.right_only_columns = sorted(set(df2.columns) - set(df1.columns))

    if index_cols:
        print("Indexing dataframes ...")
        df1 = df1.sort_values(index_cols)
        df1 = df1.set_index(index_cols)
        df2 = df2.sort_values(index_cols)
        df2 = df2.set_index(index_cols)

        print("Validating dataframe indices ...")
        l_dup_indices = df1.index.duplicated(keep=False)
        r_dup_indices = df2.index.duplicated(keep=False)
        if l_dup_indices.any():
            # Consider this an error because we are unable to determine if the data matches
            result.match = False
            result.left_index_duplicates = pd.DataFrame(
                df1.index[l_dup_indices].value_counts().rename("count")
            )
            df1 = df1[~df1.index.duplicated(keep="first")]
        if r_dup_indices.any():
            result.match = False
            result.right_index_duplicates = pd.DataFrame(
                df2.index[r_dup_indices].value_counts().rename("count")
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

    # Check if dataframes are equal (depending on tolerance options)
    if atol > 0 or rtol > 0:
        frames_equal = True
        try:
            pd.testing.assert_frame_equal(
                df1[common_cols],
                df2[common_cols],
                atol=atol,
                rtol=rtol,
                check_exact=False,
            )
        except AssertionError:
            frames_equal = False
    else:
        frames_equal = df1[common_cols].equals(df2[common_cols])

    if not frames_equal:
        result.match = False
        result.data_match = False

        for idx, col in enumerate(common_cols):
            print(f"Comparing column ({idx + 1}/{len(common_cols)}): {col} ...")
            if not df1[col].equals(df2[col]):

                mismatch_idx = series_nonequal_index(df1[col], df2[col], atol=atol, rtol=rtol)
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

                if (
                    pd.api.types.is_numeric_dtype(df1[col].dtype)
                    and pd.api.types.is_numeric_dtype(df2[col].dtype)
                    and not pd.api.types.is_bool_dtype(df1[col].dtype)
                    and not pd.api.types.is_bool_dtype(df2[col].dtype)
                ):
                    abs_diff = (df1[col] - df2[col]).abs()
                    result.column_results[col].diff_min = abs_diff.min()
                    result.column_results[col].diff_max = abs_diff.max()
                    result.column_results[col].diff_mean = abs_diff.mean()
                    result.column_results[col].diff_std = abs_diff.std()

                    # Ratio for all values (inlcuding zeros; min/max may be 0/inf)
                    ratio = (df1[col] / df2[col])
                    result.column_results[col].ratio_min = ratio.min()
                    result.column_results[col].ratio_max = ratio.max()
                    idx_inf = np.isinf(ratio)
                    result.column_results[col].ratio_mean = ratio[~idx_inf].mean()
                    result.column_results[col].ratio_std = ratio[~idx_inf].std()
                    result.column_results[col].ratio_inf_count = idx_inf.sum()

                    # Get ratio for non-zero values only
                    idx_gt_zero = (df1[col] != 0) & (df2[col] != 0)
                    ratio = (df1[col][idx_gt_zero] / df2[col][idx_gt_zero])
                    result.column_results[col].ratio_min_nonzero = ratio.min()
                    result.column_results[col].ratio_max_nonzero = ratio.max()

                    mismatch_df["absdiff"] = (
                        mismatch_df[f"{col}_LEFT"] - mismatch_df[f"{col}_RIGHT"]
                    ).abs()
                    mismatch_df = mismatch_df.sort_values("absdiff", na_position="first")
                    mismatch_df = mismatch_df.drop(columns=["absdiff"])
                    result.column_results[col].mismatch_data = mismatch_df
    return result
