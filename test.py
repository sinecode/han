import argparse

import torch
from gensim.models import KeyedVectors

from data_iterator import DataIterator
from han import Han
from utils import load_pickled_obj
import config


def main():
    parser = argparse.ArgumentParser(description="Test the HAN model")
    parser.add_argument(
        "tokenized_dataset",
        help="Pickle file where is stored the tokenized text",
    )
    parser.add_argument(
        "embedding_file", help="File name where is stored the word2vec model",
    )
    parser.add_argument(
        "model_file", help="File name where is stored the tested model",
    )

    args = parser.parse_args()

    dataset = load_pickled_obj(args.tokenized_dataset)
    wv = KeyedVectors.load(args.embedding_file)
    model_file = args.model_file
    model = Han(embedding_matrix=wv.vectors, num_classes=5)
    data_iterator = DataIterator(dataset, wv.vocab)
    test(model, data_iterator)


def test(model, data_iterator):
    model.to(config.DEVICE)
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (labels, features) in enumerate(data_iterator):
            labels = torch.LongTensor(labels)
            labels -= 1
            features = torch.LongTensor(features)

            labels = labels.to(config.DEVICE)  # why?
            features.to(config.DEVICE)

            model.init_hidden_state()

            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy: {100 * correct / total}")


if __name__ == "__main__":
    main()
