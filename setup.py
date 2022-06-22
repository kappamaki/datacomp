#!/usr/bin/env python3
from distutils.core import setup

setup(
    name="datacomp",
    version="0.4.0",
    description="CLI for comparing tabular data files",
    author="kappamaki",
    install_requires=["pandas", "pyarrow"],
    python_requires=">=3.6",
    packages=["datacomp"],
    entry_points = {"console_scripts": ["datacomp=datacomp.__main__:main"]},
)
