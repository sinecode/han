"""Train an unsupervised word2vec model"""


from statistics import mean
from typing import List
import sys

from gensim.models import Word2Vec
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm


def tokenize_text(text: str) -> List[List[str]]:
    return [word_tokenize(sentence) for sentence in sent_tokenize(text)]


class SentencesIterator:
    def __init__(self, text_df_column):
        self.text_column = text_df_column
        self.num_doc = len(text_df_column)
        self.num_sentences_per_doc = []
        self.num_words_per_doc = []

    def __iter__(self):
        for text in tqdm(self.text_column):
            sentences = tokenize_text(text.lower())
            self.num_sentences_per_doc.append(len(sentences))
            num_words = 0
            for sentence in sentences:
                num_words += len(sentence)
                yield sentence
            self.num_words_per_doc.append(num_words)

    def reset_stats(self):
        self.num_sentences_per_doc = []
        self.num_words_per_doc = []

    def get_dataset_stats(self):
        return {
            "Documents": self.num_doc,
            "Average #s": mean(self.num_sentences_per_doc),
            "Max #s": max(self.num_sentences_per_doc),
            "Average #w": mean(self.num_words_per_doc),
            "Max #w": max(self.num_words_per_doc),
        }


def train_word2vec_model(dataset_file: str):
    df = pd.read_csv(dataset_file).fillna("")
    model = Word2Vec(min_count=6, size=200)
    sentences_iter = SentencesIterator(df.text)
    model.build_vocab(sentences_iter)
    for stat_name, stat_value in sentences_iter.get_dataset_stats().items():
        print(f"{stat_name}: {stat_value}")
    print(f"Vocabulary size: {len(model.wv.vocab)}")
    sentences_iter.reset_stats()
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
