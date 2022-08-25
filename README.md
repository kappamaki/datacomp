# datacomp

`datacomp` is a python command-line tool for comparing 2 tabular data files (`.csv` and `.parquet` formats currently supported)
and providing detailed information on the similarity of the data.

By default, it will compare rows based on the row number. You can specify one or more columns to use as an index for comparing rows.

e.g.
```sh
# Compare files using columns named "id" and "timestamp" to index the rows
# (will raise an error if these columns do not exist in either input file)
datacomp filepath1.parquet filepath2.parquet id timestamp
```

For more detailed usage instructions, run `datacomp --help`.
