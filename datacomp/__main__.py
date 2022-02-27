#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import pandas as pd

from datacomp.compare import CompareResult, compare_data


CRED = "\033[91m"
CGRN = "\033[92m"
CEND = "\033[0m"


def series_nonequal_index(ser1, ser2):
    ser1_null = ser1.isnull()
    ser2_null = ser2.isnull()
    # Get index where series values are null in one series but the other
    nonequal_null_idx = list(set(ser1[ser1_null].index) ^ set(ser2[ser2_null].index))
    # Get index where series non-null values are nonequal
    notnull_idx = set(ser1[~ser1_null].index) & set(ser2[~ser2_null].index)
    nonequal_notnull_idx = ser1.loc[notnull_idx] != ser2.loc[notnull_idx]
    nonequal_notnull_idx = ser1.loc[notnull_idx][nonequal_notnull_idx].index.tolist()
    # Combine null/non-null indexes
    return nonequal_null_idx + nonequal_notnull_idx


def load_file(fpath):
    if fpath.suffix == ".parquet":
        return pd.read_parquet(fpath)
    if fpath.suffix == ".csv":
        return pd.read_csv(fpath)

    raise RuntimeError(f"Unsupported file type {fpath.suffix} for {fpath}")


def series_to_str(series, **kwargs):
    if "max_rows" not in kwargs:
        kwargs["max_rows"] = pd.get_option("display.max_rows")
    if "min_rows" not in kwargs:
        kwargs["min_rows"] = pd.get_option("display.min_rows")

    return series.to_string(**kwargs)


def dataframe_to_str(df, **kwargs):
    if "max_rows" not in kwargs:
        kwargs["max_rows"] = pd.get_option("display.max_rows")
    if "min_rows" not in kwargs:
        kwargs["min_rows"] = pd.get_option("display.min_rows")
    if "max_colwidth" not in kwargs:
        kwargs["max_colwidth"] = pd.get_option("display.max_colwidth")

    return df.to_string(**kwargs)


def header(msg):
    hbar = "=" * len(msg)
    return f"\n{hbar}\n{msg}\n{hbar}\n"


def print_result(result: CompareResult):
    if result.left_only_columns or result.right_only_columns:
        diff_msg = header("COLUMNS DO NOT MATCH")
        if result.left_only_columns:
            diff_msg += "left-only columns:\n"
            for col in result.left_only_columns:
                diff_msg += f" {col}\n"
        if result.left_only_columns and result.right_only_columns:
            diff_msg += "\n"
        if result.right_only_columns:
            diff_msg += "right-only columns:\n"
            for col in result.right_only_columns:
                diff_msg += f" {col}\n"

        print(CRED + diff_msg + CEND)

        if result.left_index_duplicates is not None or result.right_index_duplicates is not None:
            diff_msg = header("DATA CONTAINS DUPLICATE INDICES")

            def dup_indices_msg(side, dup_index_data):
                dup_index_count = len(dup_index_data)
                msg = (
                    f"{side} data contains {dup_index_count} rows with duplicate indices "
                    "(dropped):\n"
                )
                msg += dataframe_to_str(dup_index_data)
                msg += "\n"
                return msg

            if result.left_index_duplicates is not None:
                diff_msg += dup_indices_msg("left", result.left_index_duplicates)
            if (
                result.left_index_duplicates is not None
                and result.right_index_duplicates is not None
            ):
                diff_msg += "\n"
            if result.right_index_duplicates is not None:
                diff_msg += dup_indices_msg("right", result.right_index_duplicates)

            print(CRED + diff_msg + CEND)

    if not result.index_match:
        diff_msg = header("INDICES DO NOT MATCH")
        diff_msg += f"left index count: {result.left_index_count}\n"
        diff_msg += f"right index count: {result.right_index_count}\n\n"

        if result.left_only_indexes is not None:
            diff_msg += (
                f"left-only indices ({len(result.left_only_indexes)}):\n"
                f"{series_to_str(result.left_only_indexes, index=None)}\n"
            )
        if (
            result.left_only_indexes is not None
            and result.right_only_indexes is not None
        ):
            diff_msg += "\n"
        if result.right_only_indexes is not None:
            diff_msg += (
                f"right-only indices ({len(result.right_only_indexes)}):\n"
                f"{series_to_str(result.right_only_indexes, index=None)}\n"
            )

        diff_msg += header(f"COMPARING {result.common_index_count} ROWS WITH COMMON IDS")
        print(CRED + diff_msg + CEND)

    if not result.data_match:
        diff_msg = header("DATA DOES NOT MATCH")
        print(CRED + diff_msg + CEND)

        for col, col_result in result.column_results.items():
            diff_msg = header(f'COLUMN "{col}" VALUES DO NOT MATCH')
            diff_msg += (
                f"{col_result.mismatch_percent:.5f}% "
                f"({col_result.mismatch_number}) of values differ\n"
            )
            diff_msg += "\nMismatched Values\n"
            diff_msg += f"{dataframe_to_str(col_result.mismatch_data)}\n"
            print(CRED + diff_msg + CEND)

    if not result.match:
        sys.exit(1)
    else:
        print(CGRN + "Files match! 🙂" + CEND)


def main():
    parser = argparse.ArgumentParser(description="compare two data files")
    parser.add_argument(
        "filepath_left",
        type=Path,
        help="path to data file",
    )
    parser.add_argument(
        "filepath_right",
        type=Path,
        help="path to data file",
    )
    parser.add_argument(
        "index",
        nargs="*",
        help="one or more columns to use as an index for the data "
             "(optional: row number will be used if no arguments given)"
    )
    args = parser.parse_args()
    result = compare_data(args.filepath_left, args.filepath_right, args.index)
    print_result(result)


if __name__ == "__main__":
    main()
