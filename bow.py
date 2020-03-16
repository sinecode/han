"""
BoW based model.

A multinomial logistic regression is used.

- Classic BoW
- BoW-TFIDF
"""


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import GridSearchCV

# Suppress warnings from LogisticRegression
import warnings
import sys
import os

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"  # Also affect subprocesses


def load_dataset(dataset_name):
    return pd.read_csv(dataset_name)


def train_test_split(df, train_size, test_size_per_class):
    """
    Split the dataset in text_train, y_train, text_test and y_test.

    The training set is composed by `train_size` samples.

    The test set is composed by `test_size_per_class` samples for
    each class.
    """
    test_df = pd.concat(
        [
            df[df.stars == star].sample(test_size_per_class, random_state=20)
            for star in df.stars.unique()
        ]
    )
    assert (
        test_df.stars.value_counts(sort=False) == [test_size_per_class] * 5
    ).all()

    train_df = df.drop(test_df.index).sample(train_size, random_state=20)
    assert (train_df.count() == [train_size] * 2).all()

    text_train, y_train, text_test, y_test = (
        train_df.text.values,
        train_df.stars.values,
        test_df.text.values,
        test_df.stars.values,
    )
    assert text_train.shape == y_train.shape
    assert text_test.shape == y_test.shape

    return text_train, y_train, text_test, y_test


def run_experiment(vectorizer, text_train, y_train, text_test, y_test):
    """
    Run the experiment with the specified vectorizer.

    Return the accuracy score on the test set.
    """
    pipe = make_pipeline(
        vectorizer, MaxAbsScaler(copy=False), LogisticRegression(solver="sag"),
    )
    param_grid = {"logisticregression__C": [0.001, 0.01, 0.1, 1, 10]}
    grid = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1)
    grid.fit(text_train, y_train)
    print(f"Best CV score: {grid.best_score_}")
    print(f"Best parameters: {grid.best_params_}")
    return grid.score(text_test, y_test)


def main():
    exp_data = [
        ("yelp.csv", 130_000, 10_000),
        ("amazon.json", 600_000, 130_000),
    ]
    for dataset_name, train_size, test_size_per_class in exp_data:
        print("==============================")
        print(dataset_name.split(".")[0])
        dataset = load_dataset(dataset_name)
        text_train, y_train, text_test, y_test = train_test_split(
            df, train_size, test_size_per_class
        )
        accuracy = run_experiment(
            CountVectorizer(max_features=50_000),
            text_train,
            y_train,
            text_test,
            y_test,
        )
        print(f"BoW: {accuracy}")
        accuracy = run_experiment(
            TfidfVectorizer(max_features=50_000, norm=None),
            text_train,
            y_train,
            text_test,
            y_test,
        )
        print(f"BoW-TFIDF: {accuracy}")


if __name__ == "__main__":
    main()
