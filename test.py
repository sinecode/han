import argparse

import torch
import pandas as pd
from gensim.models import KeyedVectors
from tqdm import tqdm

from dataset import HierarchicalDataset, FlatDataset
from models import Fan, Han
from config import (
    BATCH_SIZE,
    PADDING,
    DEVICE,
    TQDM,
    WORD_HIDDEN_SIZE,
    SENT_HIDDEN_SIZE,
    Yelp,
    Yahoo,
    Amazon,
    Synthetic,
)


def main():
    parser = argparse.ArgumentParser(description="Test the model")
    parser.add_argument(
        "dataset",
        choices=["yelp", "yahoo", "amazon", "synthetic"],
        help="Choose the dataset",
    )
    parser.add_argument(
        "model",
        choices=["fan", "han"],
        help="Choose the model to be tested (flat or hierarchical",
    )
    parser.add_argument(
        "model_file", help="File where the trained models is stored",
    )

    args = parser.parse_args()

    if args.dataset == "yelp":
        dataset_config = Yelp
    elif args.dataset == "yahoo":
        dataset_config = Yahoo
    elif args.dataset == "amazon":
        dataset_config = Amazon
    elif args.dataset == "synthetic":
        dataset_config = Synthetic
    else:
        # should not end there
        exit()

    wv = KeyedVectors.load(dataset_config.EMBEDDING_FILE)

    test_df = pd.read_csv(dataset_config.TEST_DATASET).fillna("")
    test_documents = test_df.text
    test_labels = test_df.label
    if args.model == "fan":
        test_dataset = FlatDataset(
            test_documents,
            test_labels,
            wv.vocab,
            dataset_config.WORDS_PER_DOC[PADDING],
        )
    else:
        test_dataset = HierarchicalDataset(
            test_documents,
            test_labels,
            wv.vocab,
            dataset_config.SENT_PER_DOC[PADDING],
            dataset_config.WORDS_PER_SENT[PADDING],
        )
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=True
    )

    if args.model == "fan":
        model = Fan(
            embedding_matrix=wv.vectors,
            word_hidden_size=WORD_HIDDEN_SIZE,
            num_classes=len(test_labels.unique()),
            batch_size=BATCH_SIZE,
        ).to(DEVICE)
    else:
        model = Han(
            embedding_matrix=wv.vectors,
            word_hidden_size=WORD_HIDDEN_SIZE,
            sent_hidden_size=SENT_HIDDEN_SIZE,
            num_classes=len(test_labels.unique()),
            batch_size=BATCH_SIZE,
        ).to(DEVICE)
    model.load_state_dict(torch.load(args.model_file, map_location=DEVICE))

    criterion = torch.nn.NLLLoss().to(DEVICE)
    loss, acc = test_func(model, test_data_loader, criterion)
    print(f"Loss: {loss:.4f}, Accuracy: {acc * 100:.1f}%")


def test_func(model, data_loader, criterion):
    "Return the loss and the accuracy of the model on the input dataset"
    model.eval()
    losses = []
    accs = []
    with torch.no_grad():
        for labels, features in tqdm(
            data_loader, total=len(data_loader), disable=(not TQDM)
        ):
            labels = labels.to(DEVICE)
            features = features.to(DEVICE)

            batch_size = len(labels)
            model.init_hidden_state(batch_size)

            outputs = model(features)
            loss = criterion(outputs, labels)

            losses.append(loss.item())
            accs.append(
                (outputs.argmax(1) == labels).sum().item() / batch_size
            )
    return sum(losses) / len(losses), sum(accs) / len(accs)


if __name__ == "__main__":
    main()
