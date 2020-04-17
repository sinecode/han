from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm
from nltk.tokenize import sent_tokenize, word_tokenize


def tokenize_doc(doc: str) -> List[List[str]]:
    """
    Tokenize a document (a regular Python string) into sentences and words.
    The string is converted to lowercase before tokenization.

    >>> tokenize_doc("Hello There! General Kenobi!")
    [['hello', 'there', '!'], ['general', 'kenobi', '!']]
    """
    doc = doc.lower()
    return [word_tokenize(sentence) for sentence in sent_tokenize(doc)]


def dataset_stats(dataset_path):
    "Get the statistics reported in the README"
    documents = pd.read_csv(dataset_path).fillna("").text
    sent_lengths = []
    doc_lengths = []
    for doc in tqdm(documents, total=len(documents)):
        tokenized_doc = tokenize_doc(doc)
        doc_lengths.append(len(tokenized_doc))
        for sent in tokenized_doc:
            sent_lengths.append(len(sent))
    print(f"Stats for {dataset_path}:")
    print("")
    print(f"\tAverage #s: {np.mean(doc_lengths):.1f}")
    print(f"\t80th percentile #s: {np.percentile(doc_lengths, 80)}")
    print(f"\t90th percentile #s: {np.percentile(doc_lengths, 90)}")
    print(f"\tMax #s: {np.max(doc_lengths)}")
    print("")
    print(f"\tAverage #w: {np.mean(sent_lengths):.1f}")
    print(f"\t80th percentile #w: {np.percentile(sent_lengths, 80)}")
    print(f"\t90th percentile #w: {np.percentile(sent_lengths, 90)}")
    print(f"\tMax #w: {np.max(sent_lengths)}")


if __name__ == "__main__":
    import doctest

    doctest.testmod()
