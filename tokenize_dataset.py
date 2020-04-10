import argparse
from statistics import mean
from typing import List, Tuple

import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm

from utils import save_obj_as_pickle


def tokenize_dataset(dataset_file_name, output_file_name):
    df = pd.read_csv(dataset_file_name).fillna("")
    num_documents = df.shape[0]
    num_sent_per_doc = []
    num_words_per_sent = []
    num_words_per_doc = []
    tokenized: List[Tuple[int, List[List[str]]]] = []  # dataset
    for label, doc in tqdm(df.itertuples(index=False), total=num_documents):
        doc = preprocess_doc(doc)
        sentences = []
        num_words_in_doc = 0
        for sentence in sent_tokenize(doc):
            tokens = word_tokenize(sentence)
            num_words_per_sent.append(len(tokens))
            num_words_in_doc += len(tokens)
            sentences.append(tokens)
        num_sent_per_doc.append(len(sentences))
        num_words_per_doc.append(num_words_in_doc)
        tokenized.append((label, sentences))
    assert len(num_sent_per_doc) == num_documents
    assert len(num_words_per_doc) == num_documents
    print(f"Number of documents: {num_documents}")
    print(f"Average sentences per document: {mean(num_sent_per_doc)}")
    print(f"Max sentences per document: {max(num_sent_per_doc)}")
    print(f"Average words per sentence: {mean(num_words_per_sent)}")
    print(f"Max words per sentence: {max(num_words_per_sent)}")
    print(f"Average words per document: {mean(num_words_per_doc)}")
    print(f"Max words per document: {max(num_words_per_doc)}")
    return tokenized


def sort_dataset(dataset):
    """
    Sort the input dataset by the number of sentences in the document,
    from the document with the largest number of sentences to the
    document with the minor number.
    """
    return sorted(dataset, key=lambda x: len(x[1]), reverse=True)


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
    dataset = tokenize_dataset(dataset_file_name, output_file_name)
    dataset = sort_dataset(dataset)
    save_obj_as_pickle(dataset, output_file_name)
