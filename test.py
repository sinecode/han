import argparse

import torch
import pandas as pd
from gensim.models import KeyedVectors

from data_iterator import DataIterator
from han import Han
import config


def main():
    parser = argparse.ArgumentParser(description="Test the HAN model")
    parser.add_argument(
        "test_dataset", help="CSV file where is stored the test dataset",
    )
    parser.add_argument(
        "embedding_file", help="File name where is stored the word2vec model",
    )
    parser.add_argument(
        "model_file", help="File name where is stored the trained model",
    )

    args = parser.parse_args()

    test_df = pd.read_csv(args.test_dataset).fillna("")
    wv = KeyedVectors.load(args.embedding_file)
    model = Han(
        embedding_matrix=wv.vectors, num_classes=len(test_df.label.unique())
    ).to(config.DEVICE)
    model.load_state_dict(torch.load(args.model_file))
    data_iterator = DataIterator(test_df, wv.vocab)
    criterion = torch.nn.NLLLoss().to(config.DEVICE)
    loss, acc = test_func(model, data_iterator, criterion)
    print(f"Loss: {loss:.4f}, Accuracy {acc * 100:.1f}%")


def test_func(model, data_iterator, criterion):
    "Return the loss and the accuracy of the model on the input dataset"
    model.eval()
    loss = 0.0
    acc = 0.0
    with torch.no_grad():
        for i, (labels, features) in enumerate(data_iterator):
            labels = labels.to(config.DEVICE)
            features = features.to(config.DEVICE)

            model.init_hidden_state()

            outputs = model(features)
            loss = criterion(outputs, labels)

            loss += loss.item()
            acc += (outputs.argmax(1) == labels).sum().item()
    total_it = len(data_iterator)
    total_samples = total_it * data_iterator.batch_size
    return loss.item() / total_it, acc / total_samples


if __name__ == "__main__":
    main()
