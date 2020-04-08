import argparse

import torch
from gensim.models import KeyedVectors

from data_iterator import DataIterator
from han import Han
from utils import load_pickled_obj


def main():
    parser = argparse.ArgumentParser(description="Train the HAN model")
    parser.add_argument(
        "tokenized_dataset",
        help="Pickle file where is stored the tokenized text",
    )
    parser.add_argument(
        "embedding_file", help="File name where is stored the word2vec model",
    )
    parser.add_argument(
        "model_file", help="File name where to store the trained model",
    )

    args = parser.parse_args()

    dataset = load_pickled_obj(args.tokenized_dataset)
    wv = KeyedVectors.load(args.embedding_file)
    model_file = args.model_file
    model = Han(embedding_matrix=wv.vectors, num_classes=5)
    data_iterator = DataIterator(dataset, wv.vocab)
    train(model, data_iterator)
    torch.save(model.state_dict(), model_file)


def train(model, data_iterator):
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    for epoch in range(1):
        running_loss = 0.0
        for i, (labels, features) in enumerate(data_iterator):
            labels = torch.LongTensor(labels)
            labels -= 1
            features = torch.LongTensor(features)

            optimizer.zero_grad()
            model.init_hidden_state()

            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print(
                    f"[epoch={epoch}, batch={i}] "
                    f"Training loss: {round(running_loss, 3)}"
                )


if __name__ == "__main__":
    main()
