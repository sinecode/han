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


def train_test_split(df):
    """
    Split the dataset in text_train, y_train, text_test and y_test.

    The training set is composed by 130,000 samples.

    The test set is composed by 10,000 samples for each star,
    there are a total of five stars so the test set has size 50,000.
    """
    test_df = pd.concat(
        [
            df[df.stars == star].sample(10_000, random_state=20)
            for star in df.stars.unique()
        ]
    )
    assert (test_df.stars.value_counts(sort=False) == [10_000] * 5).all()

    train_df = df.drop(test_df.index).sample(130_000, random_state=20)
    assert (train_df.count() == [130_000] * 2).all()

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
    # pipe.fit(text_train, y_train)
    # return accuracy_score(y_test, pipe.predict(text_test))
    param_grid = {"logisticregression__C": [0.001, 0.01, 0.1, 1, 10]}
    grid = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1)
    grid.fit(text_train, y_train)
    print(f"Best CV score: {grid.best_score_}")
    print(f"Best parameters: {grid.best_params_}")
    return grid.score(text_test, y_test)


def main():
    df = pd.read_csv("datasets/yelp.csv")
    assert (df.count() == [1_569_264] * 2).all()
    text_train, y_train, text_test, y_test = train_test_split(df)
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
