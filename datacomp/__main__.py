#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import pandas as pd


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


def compare_data(data_fpath_1, data_fpath_2, index_cols):
    data_matches = True

    print(f"Loading {data_fpath_1} ...")
    df1 = load_file(data_fpath_1)
    print(f"Loading {data_fpath_2} ...")
    df2 = load_file(data_fpath_2)

    if set(df1.columns) != set(df2.columns):
        data_matches = False
        diff_msg = header("COLUMNS DO NOT MATCH")
        l_cols = pd.Series(sorted(set(df1.columns) - set(df2.columns)), dtype=str)
        r_cols = pd.Series(sorted(set(df2.columns) - set(df1.columns)), dtype=str)

        if not l_cols.empty:
            diff_msg += "left-only columns:\n"
            for col in l_cols:
                diff_msg += f" {col}\n"
        if not l_cols.empty and not r_cols.empty:
            diff_msg += "\n"
        if not r_cols.empty:
            diff_msg += "right-only columns:\n"
            for col in r_cols:
                diff_msg += f" {col}\n"

        print(CRED + diff_msg + CEND)

    if index_cols:
        print("Indexing dataframes ...")
        df1 = df1.sort_values(index_cols).set_index(index_cols)
        df2 = df2.sort_values(index_cols).set_index(index_cols)

        print("Validating dataframe indices ...")
        l_dup_indices = df1.index.duplicated()
        r_dup_indices = df2.index.duplicated()
        l_dup_index_count = l_dup_indices.sum()
        r_dup_index_count = r_dup_indices.sum()

        if l_dup_index_count or r_dup_index_count:
            # Consider this an error because we are unable to determine if the data matches
            data_matches = False
            diff_msg = header(f"DATA CONTAINS DUPLICATE INDICES")

            def dup_indices_msg(side, df, dup_indices, dup_index_count):
                dup_index_df = pd.DataFrame(df.index[dup_indices].value_counts().rename("count"))
                msg = f"{side} data contains {dup_index_count} rows with duplicate indices (dropping):\n"
                msg += dataframe_to_str(dup_index_df)
                msg += "\n"
                return msg

            if l_dup_index_count:
                diff_msg += dup_indices_msg("left", df1, l_dup_indices, l_dup_index_count)
                df1 = df1.loc[~l_dup_indices]
            if l_dup_index_count and r_dup_index_count:
                diff_msg += "\n"
            if r_dup_index_count:
                diff_msg += dup_indices_msg("right", df2, r_dup_indices, r_dup_index_count)
                df2 = df2.loc[~r_dup_indices]

            print(CRED + diff_msg + CEND)

    print("Comparing dataframe indices ...")
    if not df1.index.equals(df2.index):
        data_matches = False
        diff_msg = header("INDICES DO NOT MATCH")
        diff_msg += f"left index count: {len(df1)}\n"
        diff_msg += f"right index count: {len(df2)}\n\n"
        l_indices = pd.Series(sorted(set(df1.index) - set(df2.index)), dtype=df1.index.dtype)
        r_indices = pd.Series(sorted(set(df2.index) - set(df1.index)), dtype=df2.index.dtype)

        if not l_indices.empty:
            diff_msg += f"left-only indices ({len(l_indices)}):\n{series_to_str(l_indices, index=None)}\n"
        if not l_indices.empty and not r_indices.empty:
            diff_msg += "\n"
        if not r_indices.empty:
            diff_msg += f"right-only indices ({len(r_indices)}):\n{series_to_str(r_indices, index=None)}\n"

        common_ids = set(df1.index) & set(df2.index)
        diff_msg += header(f"COMPARING {len(common_ids)} ROWS WITH COMMON IDS")
        print(CRED + diff_msg + CEND)

        if not common_ids:
            sys.exit(1)

        df1 = df1.loc[common_ids]
        df2 = df2.loc[common_ids]

    common_cols = [col for col in df1 if col in set(df2.columns)]
    # quick check if dataframes match, otherwise show detailed differences
    print("Comparing data ...")
    if not df1[common_cols].equals(df2[common_cols]):
        diff_msg = header("DATA DOES NOT MATCH")
        print(CRED + diff_msg + CEND)

        for idx, col in enumerate(common_cols):
            print(f"Comparing column ({idx + 1}/{len(common_cols)}): {col} ...")
            if not df1[col].equals(df2[col]):
                data_matches = False

                mismatch_idx = series_nonequal_index(df1[col], df2[col])
                mismatch_df = df1.loc[mismatch_idx, [col]].join(
                    df2.loc[mismatch_idx, [col]], lsuffix="_LEFT", rsuffix="_RIGHT"
                )
                mismatch_pct = len(mismatch_idx) / len(df1) * 100

                diff_msg = header(f'COLUMN "{col}" VALUES DO NOT MATCH')
                diff_msg += f"{mismatch_pct:.5f}% ({len(mismatch_df)}) of values differ\n"
                diff_msg += "\nMismatched Values\n"
                diff_msg += f"{dataframe_to_str(mismatch_df)}\n"
                print(CRED + diff_msg + CEND)

    if not data_matches:
        sys.exit(1)
    else:
        print(CGRN + "Files match! 🙂" + CEND)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fpath1", type=Path)
    parser.add_argument("fpath2", type=Path)
    parser.add_argument("index", nargs="*")
    args = parser.parse_args()
    compare_data(args.fpath1, args.fpath2, args.index)


if __name__ == "__main__":
    main()
