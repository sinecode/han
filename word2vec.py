"""Train an unsupervised word2vec model"""


import gensim
from gensim.models import Word2Vec

import utils


def train_word2vec_model(dataset_file: str) -> gensim.models.word2vec.Word2Vec:
    df = utils.load_tokenized_dataset(dataset_file)
    sentences = df.tokens.sum()
    model = Word2Vec(min_count=6, size=200)
    model.build_vocab(sentences)
    model.train(
        sentences, total_examples=model.corpus_count, epochs=model.epochs
    )
    return model


if __name__ == "__main__":
    model = train_word2vec_model("datasets/yahoo_1k_sample_tokenized.csv")
    model.save("embedding/yahoo_embedding.wv")
