import argparse

import torch
import pandas as pd
from gensim.models import KeyedVectors
from tqdm import tqdm

from dataset import SentWordDataset
from models import Han
from config import BATCH_SIZE, DEVICE, TQDM, WORD_HIDDEN_SIZE, SENT_HIDDEN_SIZE


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

    wv = KeyedVectors.load(args.embedding_file)

    test_df = pd.read_csv(args.test_dataset).fillna("")
    test_documents = test_df.text
    test_labels = test_df.label
    test_dataset = SentWordDataset(test_documents, test_labels, wv.vocab)
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=True
    )

    model = Han(
        embedding_matrix=wv.vectors,
        word_hidden_size=WORD_HIDDEN_SIZE,
        sent_hidden_size=SENT_HIDDEN_SIZE,
        num_classes=len(test_labels.unique()),
        batch_size=BATCH_SIZE,
    ).to(DEVICE)
    model.load_state_dict(torch.load(args.model_file))

    criterion = torch.nn.NLLLoss().to(DEVICE)
    loss, acc = test_func(model, test_data_loader, criterion)
    print(f"Loss: {loss:.4f}, Accuracy {acc * 100:.1f}%")


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
