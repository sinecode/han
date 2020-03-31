"""Train an unsupervised word2vec model"""


import argparse
import pickle

from gensim.models import Word2Vec
from tqdm import tqdm


def load_pickled_dataset(filename):
    with open(filename, "rb") as f:
        ds = pickle.load(f)
    return ds


class SentenceIterator:
    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        for _, doc in tqdm(self.dataset):
            for sentence in doc:
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
        "tokenized_dataset",
        help="Pickle file where is stored the tokenized text",
    )
    parser.add_argument(
        "embedding_file", help="File name where to store the word2vec model",
    )

    args = parser.parse_args()
    tokenized_dataset = args.tokenized_dataset
    embedding_file_name = args.embedding_file

    print(f"Processing {tokenized_dataset}")
    ds = load_pickled_dataset(tokenized_dataset)
    model = train_word2vec_model(ds)
    model.wv.save(embedding_file_name)
    print(f"Saved {embedding_file_name}")
