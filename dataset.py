import numpy as np
import torch

from config import SENT_PER_DOC, WORDS_PER_SENT
from utils import tokenize_doc


class SentWordDataset(torch.utils.data.Dataset):
    """
    Dataset where each document is tokenized on the fly.

    Each element is a tuple (label, [sentences, words])
    """

    def __init__(self, documents, labels, vocab):
        super(SentWordDataset, self).__init__()
        assert len(documents) == len(labels)
        self.documents = documents
        self.labels = labels
        self.vocab = vocab

    def _word_to_index(self, word):
        try:
            return self.vocab[word].index
        except KeyError:
            return self.vocab["UNK"].index  # Out-Of-Vocabulary (OOV) word

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, index):
        label = self.labels[index]
        doc = self.documents[index]
        features = np.zeros(
            shape=(SENT_PER_DOC, WORDS_PER_SENT), dtype=np.int64
        )
        for i, sent in zip(range(SENT_PER_DOC), tokenize_doc(doc)):
            for j, word in zip(range(WORDS_PER_SENT), sent):
                features[i, j] = self._word_to_index(word)
        return label, features
