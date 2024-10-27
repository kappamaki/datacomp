#!/usr/bin/env python3
import argparse
import shutil
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

from datacomp.compare import CompareResult, compare_data


CRED = "\033[91m"
CGRN = "\033[92m"
CEND = "\033[0m"


def load_file(fpath: Path) -> pd.DataFrame:
    if fpath.is_file():
        if fpath.suffix == ".parquet":
            return pd.read_parquet(fpath).reset_index()
        if fpath.suffix == ".csv" or fpath.name.endswith(".csv.gz"):
            return pd.read_csv(fpath)
        raise RuntimeError(f'Unsupported file type "{fpath.suffix}" for "{fpath}"')

    elif fpath.is_dir():
        suffixes = sorted(
            set(
                ".csv" if f.name.endswith(".csv.gz") else f.suffix
                for f in fpath.iterdir()
            )
        )
        if len(suffixes) > 1:
            raise RuntimeError(f'Directory "{fpath}" contains mixed file types: {suffixes}')
        elif len(suffixes) == 0:
            raise RuntimeError(f'Directory "{fpath}" is empty')
        elif suffixes == [".parquet"]:
            return pd.read_parquet(fpath).reset_index()
        elif suffixes == [".csv"]:
            return pd.concat([pd.read_csv(f) for f in fpath.iterdir()], ignore_index=True)
        raise RuntimeError(f'Directory "{fpath}" contains unsupported file types: {suffixes[0]}')

    raise RuntimeError(f'Filepath "{fpath}" not found!')


def series_to_str(series, max_colwidth: Optional[int] = None, **kwargs):
    if "max_rows" not in kwargs:
        kwargs["max_rows"] = pd.get_option("display.max_rows")
    if "min_rows" not in kwargs:
        kwargs["min_rows"] = pd.get_option("display.min_rows")

    if max_colwidth is None:
        max_colwidth = shutil.get_terminal_size((80, 20)).columns  # pass fallback

    with pd.option_context("display.max_colwidth", max_colwidth):
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
            msg_index_word = "index" if dup_index_count > 0 else "indices"
            msg = (
                f"{side} data contains {dup_index_count} duplicated {msg_index_word}\n"
                "(dropped duplicate rows):\n"
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
        diff_msg += f"right index count: {result.right_index_count}\n"
        diff_msg += f"commmon index count: {result.common_index_count}\n\n"

        if not result.left_only_indexes.empty:
            diff_msg += (
                f"left-only indices ({len(result.left_only_indexes)}):\n"
                f"{series_to_str(result.left_only_indexes, index=None)}\n"
            )
        if not result.left_only_indexes.empty and not result.right_only_indexes.empty:
            diff_msg += "\n"
        if not result.right_only_indexes.empty:
            diff_msg += (
                f"right-only indices ({len(result.right_only_indexes)}):\n"
                f"{series_to_str(result.right_only_indexes, index=None)}\n"
            )

        print(CRED + diff_msg + CEND)

    for col, col_result in result.column_results.items():
        if not col_result.dtype_match:
            diff_msg = header(f'COLUMN "{col}" DATA TYPES DO NOT MATCH')
            diff_msg += f"left data type: {col_result.left_dtype}\n"
            diff_msg += f"right data type: {col_result.right_dtype}\n"
            print(CRED + diff_msg + CEND)

        if col_result.mismatch_number:
            # Number formatting used for aligned numbers
            num_format = ".5g"

            diff_msg = header(f'COLUMN "{col}" VALUES DO NOT MATCH')
            diff_msg += (
                f"{col_result.mismatch_percent:{num_format}}% "
                f"({col_result.mismatch_number}) of values differ\n"
            )

            if col_result.diff_min is not None:
                diff_msg += "\n"
                max_chars = max(
                    len(f"{num:{num_format}}") for num in [
                        col_result.diff_min,
                        col_result.diff_max,
                        col_result.diff_mean,
                        col_result.diff_std,
                    ]
                )
                num_format_align_diff = f">{max_chars}{num_format}"
                max_chars = max(
                    len(f"{num:{num_format}}") for num in [
                        col_result.ratio_min_nonzero,
                        col_result.ratio_max_nonzero,
                        col_result.ratio_mean,
                        col_result.ratio_std,
                    ]
                )
                num_format_align_ratio = f">{max_chars}{num_format}"

                diff_msg += f"diff min:  {col_result.diff_min:{num_format_align_diff}}    "
                diff_msg += f"ratio min:  {col_result.ratio_min:{num_format_align_ratio}}"
                if col_result.ratio_min == 0:
                    diff_msg += f" (non-zero: {col_result.ratio_min_nonzero:{num_format}})"
                diff_msg += "\n"

                diff_msg += f"diff max:  {col_result.diff_max:{num_format_align_diff}}    "
                diff_msg += f"ratio max:  {col_result.ratio_max:{num_format_align_ratio}}"
                if np.isinf(col_result.ratio_max):
                    diff_msg += f" (non-zero: {col_result.ratio_max_nonzero:{num_format}})"
                diff_msg += "\n"

                diff_msg += f"diff mean: {col_result.diff_mean:{num_format_align_diff}}    "
                diff_msg += f"ratio mean: {col_result.ratio_mean:{num_format_align_ratio}}"
                if col_result.ratio_inf_count > 0:
                    diff_msg += f' (exluding {col_result.ratio_inf_count} "inf" values)'
                diff_msg += "\n"

                diff_msg += f"diff std:  {col_result.diff_std:{num_format_align_diff}}    "
                diff_msg += f"ratio std:  {col_result.ratio_std:{num_format_align_ratio}}"
                if col_result.ratio_inf_count > 0:
                    diff_msg += f' (exluding {col_result.ratio_inf_count} "inf" values)'
                diff_msg += "\n"

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
    parser.add_argument(
        "--atol",
        type=float,
        default=0.0,
        help="Optional absolute tolerance to use when comparing numeric values",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=0.0,
        help="Optional relative tolerance to use when comparing numeric values",
    )

    column_filter_group = parser.add_mutually_exclusive_group()
    column_filter_group.add_argument(
        "-c",
        "--column",
        dest="columns",
        action="append",
        help=(
            "Columns to compare. Can specify more than one option (e.g. -c col1 -c col2). "
            "Mutually exclusive with the \"--exclude-column\" option."
        ),
    )
    column_filter_group.add_argument(
        "-x",
        "--exclude-column",
        dest="exclude_columns",
        action="append",
        help=(
            "Columns to exclude from comparison. "
            "Can specify more than one option (e.g. -x col1 -x col2). "
            "Mutually exclusive with the \"--column\" option."
        ),
    )

    args = parser.parse_args()

    print(f"Loading {args.filepath_left} ...")
    df1 = load_file(args.filepath_left)
    print(f"Loading {args.filepath_right} ...")
    df2 = load_file(args.filepath_right)

    for data, data_fpath in [
        (df1, args.filepath_left),
        (df2, args.filepath_right),
    ]:
        for index_col in args.index:
            if index_col not in data:
                print()
                print(
                    f'⚠️  {CRED}ERROR: index column "{index_col}" not found in {data_fpath}{CEND}'
                )
                sys.exit(1)

        if args.columns:
            for col in args.columns:
                if col not in data:
                    print()
                    print(
                        f'⚠️  {CRED}ERROR: column "{col}" not found in {data_fpath}{CEND}'
                    )
                    sys.exit(1)

    if args.exclude_columns:
        for col in args.exclude_columns:
            if col not in df1 and col not in df2:
                print()
                print(
                    f'⚠️  {CRED}ERROR: exclude column "{col}" not found in either input file{CEND}'
                )
                sys.exit(1)

    if args.columns:
        df1 = df1[args.index + args.columns]
        df2 = df2[args.index + args.columns]

    if args.exclude_columns:
        df1 = df1.drop(columns=[col for col in args.exclude_columns if col in df1])
        df2 = df2.drop(columns=[col for col in args.exclude_columns if col in df2])

    result = compare_data(df1, df2, args.index, atol=args.atol, rtol=args.rtol)

    print_result(result)


if __name__ == "__main__":
    main()
