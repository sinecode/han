import argparse

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import SGDClassifier

from config import Yelp, YelpSample, Yahoo, Amazon


def run_experiment(vectorizer, text_train, y_train, text_test, y_test):
    """
    Run the experiment with the specified vectorizer.

    Return the accuracy score on the test set.
    """
    pipe = make_pipeline(
        vectorizer,
        MaxAbsScaler(),
        SGDClassifier(loss="log", random_state=20),  # logistic regression
    )
    param_grid = {"sgdclassifier__alpha": [0.00001, 0.0001, 0.001, 0.01, 0.1]}
    grid = GridSearchCV(pipe, param_grid, cv=5, n_jobs=4)
    grid.fit(text_train, y_train)
    print(f"Best training score: {grid.best_score_}")
    print(f"Best params: {grid.best_params_}")
    return grid.score(text_test, y_test)


def process_dataset(csv_file):
    print(f"Processing {csv_file}")
    print("")
    df = pd.read_csv(csv_file).fillna("")
    text_train, text_test, y_train, y_test = train_test_split(
        df.text, df.label, test_size=0.1, stratify=df.label, random_state=20,
    )
    accuracy = run_experiment(
        CountVectorizer(max_features=50_000),
        text_train,
        y_train,
        text_test,
        y_test,
    )
    print(f"BoW: {accuracy}")
    print("")
    accuracy = run_experiment(
        TfidfVectorizer(max_features=50_000, norm=None),
        text_train,
        y_train,
        text_test,
        y_test,
    )
    print(f"BoW-TFIDF: {accuracy}")


def main():
    parser = argparse.ArgumentParser(
        description="Run the experiment with a BoW model"
    )
    parser.add_argument(
        "dataset",
        choices=["yelp", "yelp-sample", "yahoo", "amazon"],
        help="Choose the dataset",
    )

    args = parser.parse_args()

    if args.dataset == "yelp":
        dataset_config = Yelp
    elif args.dataset == "yelp-sample":
        dataset_config = YelpSample
    elif args.dataset == "yahoo":
        dataset_config = Yahoo
    elif args.dataset == "amazon":
        dataset_config = Amazon
    else:
        # should not end there
        exit()

    args = parser.parse_args()
    process_dataset(dataset_config.FULL_DATASET)


if __name__ == "__main__":
    main()
