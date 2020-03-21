"""
BoW based models.

A multinomial logistic regression is used together with
a Stochastic Gradient Descent

- Classic BoW
- BoW-TFIDF
"""


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score


DATASETS_DIR = "datasets"


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
    pipe.fit(text_train, y_train)
    return accuracy_score(y_test, pipe.predict(text_test))


def main():
    print("=================== Yelp ========================")
    print("")
    yelp_df = pd.read_csv(f"{DATASETS_DIR}/yelp.csv").fillna("")
    text_train, text_test, y_train, y_test = train_test_split(
        yelp_df.text,
        yelp_df.stars,
        test_size=0.1,
        stratify=yelp_df.stars,
        random_state=20,
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

    print("=================== Yahoo =======================")
    print("")
    yahoo_df = pd.read_csv(f"{DATASETS_DIR}/yahoo.csv").fillna("")
    text_train, text_test, y_train, y_test = train_test_split(
        yahoo_df.text,
        yahoo_df.category,
        test_size=0.1,
        stratify=yahoo_df.category,
        random_state=20,
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

    print("=================== Amazon =======================")
    print("")
    amazon_df = pd.read_csv(f"{DATASETS_DIR}/amazon.csv").fillna("")
    text_train, text_test, y_train, y_test = train_test_split(
        amazon_df.reviewText,
        amazon_df.overall,
        test_size=0.1,
        stratify=amazon_df.overall,
        random_state=20,
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
