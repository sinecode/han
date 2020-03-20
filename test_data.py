"""
Python script to test the datasets.

- yelp.csv
- yahoo.csv
- amazon.csv

Each dataset must have the same number of elements as reported in
the paper and the classes must be balanced.
"""


import pandas as pd


DATASETS_DIR = "datasets"


assert (
    pd.read_csv(f"{DATASETS_DIR}/yelp.csv").stars.value_counts()
    == [1_569_265 // 5] * 5
).all()

assert (
    pd.read_csv(f"{DATASETS_DIR}/yahoo.csv").category.value_counts()
    == [1_450_000 // 10] * 10
).all()

assert (
    pd.read_csv(f"{DATASETS_DIR}/amazon.csv").overall.value_counts()
    == [3_650_000 // 5] * 5
).all()
