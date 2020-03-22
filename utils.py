from itertools import chain
from collections import Counter
from ast import literal_eval

import pandas as pd


def split_text(nlp_pipeline, text):
    """Split the text in sentences and tokenize each sentence.

    >>> import stanza
    >>> stanza.download(lang="en", processors="tokenize", verbose=False)
    >>> nlp = stanza.Pipeline(lang="en", processors="tokenize", verbose=False)
    >>> split_text(nlp, "First sentence. Second sentence.")
    [['First', 'sentence', '.'], ['Second', 'sentence', '.']]
    """
    doc = nlp_pipeline(text)
    return [
        [token.text for token in sentence.tokens] for sentence in doc.sentences
    ]


def load_tokenized_dataset(csv_file):
    df = pd.read_csv(csv_file, dtype={"class": str, "tokens": str})
    df.tokens = df.tokens.apply(literal_eval)
    assert (df.tokens.apply(type).unique() == [list]).all()
    return df


def flat_nested_list(x):
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
