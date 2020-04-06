import numpy as np
from tqdm import tqdm


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
    def __init__(self, dataset, vocab, batch_size=64):
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
            num_doc = self.batch_size
            documents = [doc for _, doc in batch]
            max_sent = max_sent_per_doc(documents)
            max_words = max_words_per_sent(documents)
            labels = np.int8([label for label, _ in batch])
            data = np.zeros(shape=(num_doc, max_sent, max_words))
            for i, doc in enumerate(documents):
                for j, sent in zip(range(max_sent), doc):
                    for k, word in zip(range(max_words), sent):
                        data[i, j, k] = self._word_to_index(word)
            yield labels, data


if __name__ == "__main__":
    import doctest

    doctest.testmod()
