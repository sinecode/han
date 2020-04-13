import numpy as np
import pandas as pd
import torch

from config import MAX_SENT, MAX_WORDS
from utils import tokenize_doc


class MyDataset(torch.utils.data.Dataset):
    "Dataset where each document is tokenized on the fly"

    def __init__(self, dataset_path, vocab):
        super(MyDataset, self).__init__()
        self.df = pd.read_csv(dataset_path).fillna("")
        self.vocab = vocab

    def _word_to_index(self, word):
        try:
            return self.vocab[word].index
        except KeyError:
            return self.vocab["UNK"].index  # Out-Of-Vocabulary (OOV) word

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        label, doc = self.df.iloc[index]
        features = np.zeros(shape=(MAX_SENT, MAX_WORDS), dtype=np.int64)
        for i, sent in zip(range(MAX_SENT), tokenize_doc(doc)):
            for j, word in zip(range(MAX_WORDS), sent):
                features[i, j] = self._word_to_index(word)
        return label, features
