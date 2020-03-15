"""
Create the Yelp dataset used in the experiments.

This script assumes that in the `dataset` directory there is a file
called `yelp_full.json` that contains 1,569,264 reviews (lines).

See README for more info about how to get the `yelp_full.json` file.

ATTENTION: this script needs >= 16 GB of RAM.
"""


import pandas as pd


df = pd.read_json(
    "datasets/yelp_full.json", orient="records", lines=True
)

df.drop(
    columns=[c for c in df.columns if c not in ["stars", "text"]],
    inplace=True,
)
assert (df.columns == ["stars", "text"]).all()

df.stars = df.stars - 1
assert ((df.stars >= 0) & (df.stars <= 4)).all()

df.to_csv("datasets/yelp.csv", index=False)
