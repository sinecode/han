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
        self.first_time = True

    def __iter__(self):
        max_sent_in_doc = 0
        max_words_in_sent = 0
        for text in tqdm(self.text_column):
            sentences = tokenize_text(text)
            max_sent_in_doc = max(len(sentences), max_sent_in_doc)
            for sentence in sentences:
                max_words_in_sent = max(len(sentence), max_words_in_sent)
                yield sentence
        if self.first_time:
            print(f"Max number of sentences in a doc: {max_sent_in_doc}")
            print(f"Max number of words in a sentence: {max_words_in_sent}")
            self.first_time = False


def train_word2vec_model(dataset_file: str, min_count=5, dim_embedding=200):
    df = pd.read_csv(dataset_file).fillna("")
    model = Word2Vec(min_count=min_count, size=dim_embedding)
    model.build_vocab([["PAD"] * min_count])  # add PAD word
    model.build_vocab([["UNK"] * min_count], update=True)  # add OOV word
    sentences_iter = SentencesIterator(df.text)
    model.build_vocab(sentences_iter, update=True)
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
