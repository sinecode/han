import numpy as np
import torch

from utils import sent_word_tokenize, tokenize, word_to_index


class WordDataset(torch.utils.data.Dataset):
    """
    Dataset where each document is tokenized on the fly.
    The tokenization is in words only.
    """

    def __init__(self, documents, labels, vocab, words_per_doc):
        super(WordDataset, self).__init__()
        assert len(documents) == len(labels)
        self.documents = documents
        self.labels = labels
        self.vocab = vocab
        self.words_per_doc = words_per_doc

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, index):
        "Return a tuple (label, [tokens])"
        label = self.labels[index]
        doc = self.documents[index]
        features = np.zeros(shape=(self.words_per_doc), dtype=np.int64)
        for i, word in zip(range(self.words_per_doc), tokenize(doc)):
            features[i] = word_to_index(word, self.vocab)
        return label, features


class SentWordDataset(WordDataset):
    """
    Dataset where each document is tokenized on the fly.
    The tokenization is in sentences and words.
    """

    def __init__(self, documents, labels, vocab, sent_per_doc, words_per_sent):
        super(SentWordDataset, self).__init__(documents, labels, vocab, 0)
        self.sent_per_doc = sent_per_doc
        self.words_per_sent = words_per_sent

    def __getitem__(self, index):
        "Return a tuple (label, [sentences])"
        label = self.labels[index]
        doc = self.documents[index]
        features = np.zeros(
            shape=(self.sent_per_doc, self.words_per_sent), dtype=np.int64
        )
        for i, sent in zip(range(self.sent_per_doc), sent_word_tokenize(doc)):
            for j, word in zip(range(self.words_per_sent), sent):
                features[i, j] = word_to_index(word, self.vocab)
        return label, features
