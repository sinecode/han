import argparse

import pandas as pd
from gensim.models import Word2Vec
from tqdm import tqdm

from utils import tokenize_doc


class SentenceIterator:
    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        for _, doc in tqdm(
            self.dataset.itertuples(index=False), total=len(self.dataset)
        ):
            tokenized_doc = tokenize_doc(doc)
            for sentence in tokenized_doc:
                yield sentence


def train_word2vec_model(dataset, min_count=5, dim_embedding=200):
    model = Word2Vec(min_count=min_count, size=dim_embedding)
    model.build_vocab([["PAD"] * min_count])  # add PAD word
    model.build_vocab([["UNK"] * min_count], update=True)  # add OOV word
    sentence_iter = SentenceIterator(dataset)
    model.build_vocab(sentence_iter, update=True)
    model.train(
        sentence_iter, total_examples=model.corpus_count, epochs=model.epochs,
    )
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a word2vec model")
    parser.add_argument(
        "dataset", help="CSV file where is stored the dataset",
    )
    parser.add_argument(
        "embedding_file", help="File name where to store the word2vec model",
    )

    args = parser.parse_args()

    dataset = pd.read_csv(args.dataset).fillna("")
    model = train_word2vec_model(dataset)
    model.wv.save(args.embedding_file)
