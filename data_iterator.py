import numpy as np
from tqdm import tqdm


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF
    args = [iter(iterable)] * n
    return zip(*args)


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
            max_sent_per_doc = max(len(doc) for _, doc in batch)
            max_words_per_sent = max(
                len(sent) for _, doc in batch for sent in doc
            )
            labels = np.zeros(shape=(num_doc))
            data = np.zeros(
                shape=(num_doc, max_sent_per_doc, max_words_per_sent)
            )
            for i, (label, doc) in enumerate(batch):
                labels[i] = label
                for j, sent in enumerate(doc):
                    for k, word in enumerate(sent):
                        data[i, j, k] = self._word_to_index(word)
            yield labels, data
