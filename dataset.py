import numpy as np
import torch

from config import SENT_PER_DOC, WORDS_PER_SENT, WORDS_PER_DOC
from utils import sent_word_tokenize, tokenize


class WordDataset(torch.utils.data.Dataset):
    """
    Dataset where each document is tokenized on the fly.
    The tokenization is in words only.
    """

    def __init__(self, documents, labels, vocab):
        super(WordDataset, self).__init__()
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
        "Return a tuple (label, [tokens])"
        label = self.labels[index]
        doc = self.documents[index]
        features = np.zeros(shape=(WORDS_PER_DOC), dtype=np.int64)
        for i, word in zip(range(WORDS_PER_DOC), tokenize(doc)):
            features[i] = self._word_to_index(word)
        return label, features


class SentWordDataset(torch.utils.data.Dataset):
    """
    Dataset where each document is tokenized on the fly.
    The tokenization is in sentences and words.
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
        "Return a tuple (label, [sentences])"
        label = self.labels[index]
        doc = self.documents[index]
        features = np.zeros(
            shape=(SENT_PER_DOC, WORDS_PER_SENT), dtype=np.int64
        )
        for i, sent in zip(range(SENT_PER_DOC), sent_word_tokenize(doc)):
            for j, word in zip(range(WORDS_PER_SENT), sent):
                features[i, j] = self._word_to_index(word)
        return label, features
