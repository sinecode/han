"""Train an unsupervised word2vec model"""


import sys

from gensim.models import Word2Vec
import pandas as pd
from tqdm import tqdm

from utils import tokenize_text


class SentencesIterator:
    def __init__(self, text_df_column):
        self.text_column = text_df_column
        self.num_doc = len(text_df_column)

    def __iter__(self):
        for text in tqdm(self.text_column):
            sentences = tokenize_text(text)
            for sentence in sentences:
                yield sentence


def train_word2vec_model(dataset_file: str, min_count=6):
    df = pd.read_csv(dataset_file).fillna("")
    model = Word2Vec(min_count=min_count, size=200)
    sentences_iter = SentencesIterator(df.text)
    model.build_vocab(sentences_iter)
    model.build_vocab([["UNK"] * min_count], update=True)  # add OOV word
    print(f"Vocabulary size: {len(model.wv.vocab)}")
    model.train(
        sentences_iter, total_examples=model.corpus_count, epochs=model.epochs,
    )
    return model


if __name__ == "__main__":
    dataset_file_name = sys.argv[1]
    embedding_file_name = sys.argv[2]
    print(f"Processing {dataset_file_name}")
    model = train_word2vec_model(dataset_file_name)
    model.wv.save(embedding_file_name)
    print(f"Saved {embedding_file_name}")
