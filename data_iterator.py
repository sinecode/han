import numpy as np
from tqdm import tqdm
from nltk.tokenize import sent_tokenize, word_tokenize

import config
from utils import tokenize_doc


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF
    args = [iter(iterable)] * n
    return zip(*args)


def max_sent_per_doc(documents, perc=0.8):
    """Get the length of the longest document.

    The perc paramer is the percentage of documents that has length
    minor or equal to the returned value.

    >>> docs = [["a", "b", "c"], ["d", "e"], ["f", "g"]]
    >>> max_sent_per_doc(docs, perc=1)
    3
    >>> max_sent_per_doc(docs)
    2
    """
    assert 0 <= perc <= 1
    lengths = [len(doc) for doc in documents]
    return sorted(lengths)[int(perc * len(lengths)) - 1]


def max_words_per_sent(documents, perc=0.8):
    """Get the length of the longest sentences.

    The perc paramer is the percentage of sentences that has length
    minor or equal to the returned value.

    >>> docs = [[["a", "b", "c"], ["d", "e"]], [["f"]]]
    >>> max_words_per_sent(docs, perc=1)
    3
    >>> max_words_per_sent(docs)
    2
    """
    assert 0 <= perc <= 1
    lengths = [len(sent) for doc in documents for sent in doc]
    return sorted(lengths)[int(perc * len(lengths)) - 1]


class DataIterator:
    def __init__(self, dataset, vocab, batch_size=config.BATCH_SIZE):
        self.dataset = dataset
        self.vocab = vocab
        self.batch_size = batch_size

    def _word_to_index(self, word):
        try:
            return self.vocab[word].index
        except KeyError:
            return self.vocab["UNK"].index

    def __iter__(self):
        for batch in tqdm(
            grouper(self.dataset, self.batch_size),
            total=len(self.dataset),
            disable=True,
        ):
            documents = [doc for _, doc in batch]
            max_sent = max_sent_per_doc(documents)
            max_words = max_words_per_sent(documents)
            labels = np.int8([label for label, _ in batch])
            data = np.zeros(shape=(self.batch_size, max_sent, max_words))
            for i, doc in enumerate(documents):
                for j, sent in zip(range(max_sent), doc):
                    for k, word in zip(range(max_words), sent):
                        data[i, j, k] = self._word_to_index(word)
            yield labels, data


def sort_df_by_text_len(df):
    """
    Sort the input DataFrame by text length, in decreasing order.
    """
    index_sorted = df.text.str.len().sort_values(ascending=False).index
    df_sorted = df.reindex(index_sorted)
    return df_sorted.reset_index(drop=True)


class LazyDataIterator(DataIterator):
    """
    Data iterator where each document is tokenized on the fly.

    This can be useful when there's not enough RAM in the system.
    The con is that it's much slower then the DataIterator, which
    uses a pre-tokenized dataset.
    """

    def __init__(self, dataset, vocab, batch_size=config.BATCH_SIZE):
        super(LazyDataIterator, self).__init__(
            dataset=None, vocab=vocab, batch_size=batch_size
        )
        self.dataset = sort_df_by_text_len(dataset)

    def __iter__(self):
        for batch in tqdm(
            grouper(self.dataset.itertuples(index=False), self.batch_size),
            total=len(self.dataset),
            disable=True,
        ):
            labels = []
            documents = []
            for label, doc in batch:
                labels.append(label)
                documents.append(tokenize_doc(doc))
            max_sent = max_sent_per_doc(documents)
            max_words = max_words_per_sent(documents)
            labels = np.int8(labels)
            data = np.zeros(shape=(self.batch_size, max_sent, max_words))
            for i, doc in enumerate(documents):
                for j, sent in zip(range(max_sent), doc):
                    for k, word in zip(range(max_words), sent):
                        data[i, j, k] = self._word_to_index(word)
            yield labels, data


if __name__ == "__main__":
    import doctest

    doctest.testmod()
