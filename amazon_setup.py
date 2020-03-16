"""
Create the Amazon dataset used in the experiments.

This script assumes that in the `dataset` directory there is a file
called `amazon_full.json` that contains 3,650,000 reviews (lines).

See README for more info about how to get the `amazon_full.json` file.

ATTENTION: this script needs >= 16 GB of RAM.
"""


import pandas as pd


df = pd.read_json(
    "datasets/amazon_full.json", orient="records", lines=True
)

df.drop(
    columns=[c for c in df.columns if c not in ["reviewText", "overall"]],
    inplace=True,
)
df = df.rename(columns={"reviewText": "text", "overall": "stars"})
assert (df.columns == ["text", "stars"]).all()

df.stars = df.stars - 1
assert ((df.stars >= 0) & (df.stars <= 4)).all()

df.to_csv("datasets/amazon.csv", index=False)
