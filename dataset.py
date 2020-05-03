from typing import List

import numpy as np
import gensim
import torch
from nltk.tokenize import sent_tokenize, word_tokenize


def word_to_index(word: str, vocab: dict, oov_token="UNK") -> int:
    try:
        return vocab[word].index
    except KeyError:
        return vocab[oov_token].index


def document_to_flat_feature(
    doc: str, wv: gensim.models.KeyedVectors
) -> List[int]:
    tokenized_doc = word_tokenize(doc.lower())
    return [word_to_index(word, wv.vocab) for word in tokenized_doc]


def document_to_hierarchical_feature(
    doc: str, wv: gensim.models.KeyedVectors
) -> List[List[int]]:
    tokenized_doc = [
        word_tokenize(sent) for sent in sent_tokenize(doc.lower())
    ]
    return [
        [word_to_index(word, wv.vocab) for word in sent]
        for sent in tokenized_doc
    ]


def flat_feature_to_document(
    feature: List[int], wv: gensim.models.KeyedVectors
) -> str:
    return " ".join(wv.index2word[index] for index in feature)


def hierachical_feature_to_document(
    feature: List[List[int]], wv: gensim.models.KeyedVectors
) -> str:
    return " ".join(wv.index2word[index] for sent in feature for index in sent)


class FlatDataset(torch.utils.data.Dataset):
    """
    Dataset where each document is tokenized on the fly.
    The tokenization is in words only.
    """

    def __init__(self, documents, labels, vocab, words_per_doc):
        super(FlatDataset, self).__init__()
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
        tokenized_doc = word_tokenize(doc.lower())
        padded_features = np.zeros(shape=(self.words_per_doc), dtype=np.int64)
        for i, word in zip(range(self.words_per_doc), tokenized_doc):
            padded_features[i] = word_to_index(word, self.vocab)
        return label, padded_features


class HierachicalDataset(FlatDataset):
    """
    Dataset where each document is tokenized on the fly.
    The tokenization is in sentences and words.
    """

    def __init__(self, documents, labels, vocab, sent_per_doc, words_per_sent):
        super(HierachicalDataset, self).__init__(documents, labels, vocab, 0)
        self.sent_per_doc = sent_per_doc
        self.words_per_sent = words_per_sent

    def __getitem__(self, index):
        "Return a tuple (label, [sentences])"
        label = self.labels[index]
        doc = self.documents[index]
        tokenized_doc = [
            word_tokenize(sent) for sent in sent_tokenize(doc.lower())
        ]
        features = np.zeros(
            shape=(self.sent_per_doc, self.words_per_sent), dtype=np.int64
        )
        for i, sent in zip(range(self.sent_per_doc), tokenized_doc):
            for j, word in zip(range(self.words_per_sent), sent):
                features[i, j] = word_to_index(word, self.vocab)
        return label, features
