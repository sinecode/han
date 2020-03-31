import argparse
import pickle

import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm


def preprocess_doc(doc):
    return doc.lower().strip()


def tokenize_dataset(dataset_file_name, output_file_name):
    df = pd.read_csv(dataset_file_name).fillna("")
    tokenized = []
    for label, doc in tqdm(df.itertuples(index=False), total=df.shape[0]):
        doc = preprocess_doc(doc)
        sentences = [
            word_tokenize(sentence) for sentence in sent_tokenize(doc)
        ]
        tokenized.append((label, sentences))


def save_dataset(dataset, filename):
    with open(filename, "wb") as f:
        pickle.dump(tokenized, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tokenize a dataset and save the result"
    )
    parser.add_argument("dataset_file", help="The CSV file of the dataset")
    parser.add_argument(
        "output_file", help="File where to pickle the list of tokens"
    )

    args = parser.parse_args()
    dataset_file_name = args.dataset_file
    output_file_name = args.output_file
    print(f"Processing {dataset_file_name}")
    tokenized = tokenize_dataset(dataset_file_name, output_file_name)
    save_dataset(tokenized, output_file_name)
    print(f"Saved {output_file_name}")
