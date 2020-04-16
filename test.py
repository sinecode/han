import argparse

import torch
from gensim.models import KeyedVectors
from tqdm import tqdm

from dataset import MyDataset
from han import Han
from config import BATCH_SIZE, DEVICE, TQDM


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
    num_classes = 10 if "yahoo" in args.test_dataset else 5  # TODO fix
    model = Han(embedding_matrix=wv.vectors, num_classes=num_classes).to(
        DEVICE
    )
    model.load_state_dict(torch.load(args.model_file))

    test_dataset = MyDataset(args.test_dataset, wv.vocab)
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )
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
