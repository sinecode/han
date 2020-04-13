import argparse
import time

import torch
import pandas as pd
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from data_iterator import DataIterator
from han import Han
from test import test_func
import config


def main():
    parser = argparse.ArgumentParser(description="Train the HAN model")
    parser.add_argument(
        "train_dataset", help="CSV file where is stored the training dataset",
    )
    parser.add_argument(
        "val_dataset", help="CSV file where is stored the validation dataset",
    )
    parser.add_argument(
        "embedding_file", help="File name where is stored the word2vec model",
    )
    parser.add_argument(
        "model_file", help="File name where to store the trained model",
    )

    args = parser.parse_args()

    df_train = pd.read_csv(args.train_dataset).fillna("")
    df_val = pd.read_csv(args.val_dataset).fillna("")
    wv = KeyedVectors.load(args.embedding_file)
    model = Han(
        embedding_matrix=wv.vectors, num_classes=len(df_train.label.unique())
    ).to(config.DEVICE)
    train_data_iterator = DataIterator(df_train, wv.vocab)
    val_data_iterator = DataIterator(df_val, wv.vocab)
    criterion = torch.nn.NLLLoss().to(config.DEVICE)
    optimizer = torch.optim.SGD(
        (p for p in model.parameters() if p.requires_grad),
        lr=config.LEARNING_RATE,
        momentum=config.MOMENTUM,
    )
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    start_time = time.time()
    for epoch in range(1, config.EPOCHS + 1):
        train_loss, train_acc = train_func(
            model, train_data_iterator, criterion, optimizer
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_loss, val_acc = test_func(model, val_data_iterator, criterion)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        secs = int(time.time() - start_time)
        mins = secs // 60
        secs = secs % 60
        print(f"Epoch {epoch} | {mins}m {secs}s")
        print(
            f"\tTrain loss: {train_loss:.3e}, Train acc: {train_acc * 100:.1f}%"
        )
        print(f"\tVal loss: {val_loss:.3e}, Val acc: {val_acc * 100:.1f}%")
        if epoch != config.EPOCHS:
            torch.save(model.state_dict(), f"{args.model_file}-{epoch}.pth")

    plot_training(train_losses, train_accs, val_losses, val_accs)

    torch.save(model.state_dict(), f"{args.model_file}.pth")
    print("Hyperparameters:")
    print(f"\tEpochs: {config.EPOCHS}")
    print(f"\tBatch size: {config.BATCH_SIZE}")
    print(f"\tLearning rate: {config.LEARNING_RATE}")
    print(f"\tMomentum: {config.MOMENTUM}")
    print(f"\tPercentage length: {config.PERC_LENGTH}")


def train_func(model, data_iterator, criterion, optimizer):
    "Train the model and return the training loss and accuracy"
    model.train()
    loss = 0.0
    acc = 0.0
    for labels, features in data_iterator:
        labels = labels.to(config.DEVICE)
        features = features.to(config.DEVICE)

        optimizer.zero_grad()
        model.init_hidden_state()

        outputs = model(features)
        loss = criterion(outputs, labels)
        assert not torch.isnan(loss)
        loss.backward()
        optimizer.step()

        loss += loss.item()
        acc += (outputs.argmax(1) == labels).sum().item()

    total_it = len(data_iterator)
    total_samples = total_it * data_iterator.batch_size
    return loss.item() / total_it, acc / total_samples


def plot_training(train_losses, train_accs, val_losses, val_accs):
    epochs = list(range(1, len(train_losses) + 1))
    plt.figure(1)
    plt.plot(epochs, train_losses)
    plt.plot(epochs, val_losses)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.xticks(epochs)
    plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2e"))
    plt.legend(["Train", "Validation"])
    plt.tight_layout()
    plt.savefig("plots/loss.pdf")
    plt.figure(2)
    plt.plot(epochs, train_accs)
    plt.plot(epochs, val_accs)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.xticks(epochs)
    plt.legend(["Train", "Validation"])
    plt.tight_layout()
    plt.savefig("plots/accuracy.pdf")


if __name__ == "__main__":
    main()
