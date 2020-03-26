from itertools import chain
from collections import Counter
from ast import literal_eval
from typing import List

import pandas as pd
import stanza


def tokenize_text(
    nlp: stanza.pipeline.core.Pipeline, text: str,
) -> List[List[str]]:
    """
    Split the text in sentences and tokenize each sentence.
    """
    doc = nlp(text)
    return [
        [token.text.lower() for token in sentence.tokens]
        for sentence in doc.sentences
    ]


def balanced_sample(df: pd.DataFrame, num_per_class: int) -> pd.DataFrame:
    "Return a balanced sample from the input DataFrame"
    label_counts = df.label.value_counts()
    unique_labels = set(label_counts.index)
    assert (label_counts >= [num_per_class] * len(unique_labels)).all()
    return pd.concat(
        [
            df[df.label == class_label].sample(num_per_class)
            for class_label in unique_labels
        ]
    ).sample(frac=1)


def load_tokenized_dataset(csv_file: str) -> pd.DataFrame:
    df = pd.read_csv(csv_file, dtype={"class": str, "tokens": str})
    df.tokens = df.tokens.apply(literal_eval)
    assert (df.tokens.apply(type).unique() == [list]).all()
    return df


def flat_nested_list(x: List[List]) -> List:
    """Flat a list of lists to a one dimension list.

    >>> flat_nested_list([[1, 2, 3], [4, 5]])
    [1, 2, 3, 4, 5]
    >>> flat_nested_list([[1, 2, 3]])
    [1, 2, 3]
    """
    return list(chain.from_iterable(x))


def build_vocabulary(tokens, min_appearences=1):
    """Return a vocabulary (set) of the tokenized text.

    >>> sorted(build_vocabulary(['a', 'b', 'a', 'b', 'c']))
    ['a', 'b', 'c']
    >>> sorted(build_vocabulary(['a', 'b', 'a', 'b', 'c'], min_appearences=2))
    ['a', 'b']
    """
    counter = Counter(tokens)
    return set(token for token in counter if counter[token] >= min_appearences)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
