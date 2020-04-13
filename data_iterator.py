import torch
import numpy as np

import config
from utils import tokenize_doc


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF
    args = [iter(iterable)] * n
    return zip(*args)


def max_sent_per_doc(documents, perc=config.PERC_LENGTH):
    """Get the length of the longest document.

    The perc paramer is the percentage of documents that has length
    minor or equal to the returned value.

    >>> docs = [["a", "b", "c"], ["d", "e"], ["f", "g"]]
    >>> max_sent_per_doc(docs, perc=1)
    3
    >>> max_sent_per_doc(docs, perc=0.8)
    2
    """
    assert 0 <= perc <= 1
    lengths = [len(doc) for doc in documents]
    return sorted(lengths)[int(perc * len(lengths)) - 1]


def max_words_per_sent(documents, perc=config.PERC_LENGTH):
    """Get the length of the longest sentences.

    The perc paramer is the percentage of sentences that has length
    minor or equal to the returned value.

    >>> docs = [[["a", "b", "c"], ["d", "e"]], [["f"]]]
    >>> max_words_per_sent(docs, perc=1)
    3
    >>> max_words_per_sent(docs, perc=0.8)
    2
    """
    assert 0 <= perc <= 1
    lengths = [len(sent) for doc in documents for sent in doc]
    return sorted(lengths)[int(perc * len(lengths)) - 1]


def sort_df_by_text_len(df):
    "Sort the input DataFrame by text length, in decreasing order"
    index_sorted = df.text.str.len().sort_values(ascending=False).index
    df_sorted = df.reindex(index_sorted)
    return df_sorted.reset_index(drop=True)


class DataIterator:
    "Data iterator where each document is tokenized on the fly"

    def __init__(self, df, vocab, batch_size=config.BATCH_SIZE):
        self.df = sort_df_by_text_len(df)
        self.vocab = vocab
        self.batch_size = batch_size

    def _word_to_index(self, word):
        try:
            return self.vocab[word].index
        except KeyError:
            return self.vocab["UNK"].index  # OOV word

    def __iter__(self):
        for batch in grouper(self.df.itertuples(index=False), self.batch_size):
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
            yield torch.LongTensor(labels), torch.LongTensor(data)

    def __len__(self):
        return len(self.df) // self.batch_size


if __name__ == "__main__":
    import doctest

    doctest.testmod()
