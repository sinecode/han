"""Train an unsupervised word2vec model"""


from statistics import mean

import gensim
from gensim.models import Word2Vec
import pandas as pd
import stanza


class SentencesIterator:
    def __init__(self, nlp_pipeline, text_df_column):
        self.nlp_pipeline = nlp_pipeline
        self.text_column = text_df_column
        self.num_doc = len(text_df_column)
        self.num_sentences_per_doc = []
        self.num_words_per_doc = []
        self.is_cache_active = False
        self.cached_sentences = []

    def __iter__(self):
        if not self.is_cache_active:
            for text in self.text_column:
                doc = self.nlp_pipeline(text)
                self.num_sentences_per_doc.append(len(doc.sentences))
                num_words = 0
                for sentence in doc.sentences:
                    tokens = [token.text.lower() for token in sentence.tokens]
                    num_words += len(tokens)
                    self.cached_sentences.append(tokens)
                    yield tokens
                self.num_words_per_doc.append(num_words)
                if len(self.num_sentences_per_doc) % 1_000:
                    print(
                        f"Processed {len(self.num_sentences_per_doc)} of {self.num_doc} documents"
                    )
            self.is_cache_active = True
        else:
            for sentence in self.cached_sentences:
                yield sentence

    def get_dataset_stats(self):
        return {
            "Documents": self.num_doc,
            "Average #s": mean(self.num_sentences_per_doc),
            "Max #s": max(self.num_sentences_per_doc),
            "Average #w": mean(self.num_words_per_doc),
            "Max #w": max(self.num_words_per_doc),
        }


def train_word2vec_model(dataset_file: str) -> gensim.models.word2vec.Word2Vec:
    df = pd.read_csv(dataset_file).fillna("")
    model = Word2Vec(min_count=6, size=200)
    nlp = stanza.Pipeline(
        lang="en", processors="tokenize", use_gpu=True, verbose=False,
    )
    sentences_iter = SentencesIterator(nlp, df.text)
    model.build_vocab(sentences_iter)
    print(model.corpus_count)
    model.train(
        sentences_iter, total_examples=model.corpus_count, epochs=model.epochs
    )
    for stat_name, stat_value in sentences_iter.get_dataset_stats().items():
        print(f"{stat_name}: {stat_value}")
    print(f"Vocabulary size: {len(model.wv.vocab)}")
    return model


if __name__ == "__main__":
    model = train_word2vec_model("datasets/amazon.csv")
    model.wv.save("embedding/amazon_embedding.wv")
