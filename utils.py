import pickle
from typing import List

from nltk.tokenize import sent_tokenize, word_tokenize


def tokenize_doc(doc: str) -> List[List[str]]:
    """
    Tokenize a document (a regular Python string) into sentences and words.
    The string is converted to lowercase before tokenization.

    >>> tokenize_doc("Hello There! General Kenobi!")
    [['hello', 'there', '!'], ['general', 'kenobi', '!']]
    """
    doc = doc.lower().strip()
    return [word_tokenize(sentence) for sentence in sent_tokenize(doc)]


def save_obj_as_pickle(obj, filename):
    with open(filename, "wb") as f:
        pickle.dump(obj, f)


def load_pickled_obj(filename):
    with open(filename, "rb") as f:
        obj = pickle.load(f)
    return obj


if __name__ == "__main__":
    import doctest

    doctest.testmod()
