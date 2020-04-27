from typing import List, Iterable

from tqdm import tqdm
from nltk.tokenize import sent_tokenize, word_tokenize


def tokenize(doc: str) -> List[str]:
    """
    Tokenize a document (a regular Python string) into tokens.
    The string is converted to lowercase before tokenization.

    >>> tokenize("Hello There! General Kenobi!")
    ['hello', 'there', '!', 'general', 'kenobi', '!']
    """
    return word_tokenize(doc.lower())


def sent_word_tokenize(doc: str) -> List[List[str]]:
    """
    Tokenize a document (a regular Python string) into sentences and words.
    The string is converted to lowercase before tokenization.

    >>> sent_word_tokenize("Hello There! General Kenobi!")
    [['hello', 'there', '!'], ['general', 'kenobi', '!']]
    """
    doc = doc.lower()
    return [word_tokenize(sentence) for sentence in sent_tokenize(doc)]


def word_to_index(word, vocab, oov_token="UNK"):
    "Convert a word into an index according to the input vocabulary"
    try:
        return vocab[word].index
    except KeyError:
        return vocab[oov_token].index


def count_tokens(documents: Iterable[str], verbose=False):
    """Return three lists:

    - for each document, the number of sentences
    - for each sentence, the number of tokens
    - for each document, the number of tokens

    >>> count_tokens(["First document. First document again", "Second doc."])
    ([2, 1], [3, 3, 3], [6, 3])
    """
    sent_per_doc = []
    tokens_per_sent = []
    tokens_per_doc = []
    for doc in tqdm(documents, total=len(documents), disable=(not verbose)):
        tokenized_doc = sent_word_tokenize(doc)
        sent_per_doc.append(len(tokenized_doc))
        tokens = 0
        for sent in tokenized_doc:
            tokens_per_sent.append(len(sent))
            tokens += len(sent)
        tokens_per_doc.append(tokens)
    return sent_per_doc, tokens_per_sent, tokens_per_doc


if __name__ == "__main__":
    import doctest

    doctest.testmod()
