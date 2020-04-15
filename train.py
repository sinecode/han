import argparse
import time
from collections import deque

import torch
from tensorboardX import SummaryWriter
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from tqdm import tqdm

from dataset import MyDataset
from han import Han
from test import test_func
from config import BATCH_SIZE, EPOCHS, LEARNING_RATE, MOMENTUM, DEVICE


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

    wv = KeyedVectors.load(args.embedding_file)
    num_classes = 10 if "yahoo" in args.train_dataset else 5  # TODO fix
    model = Han(embedding_matrix=wv.vectors, num_classes=num_classes).to(
        DEVICE
    )

    writer = SummaryWriter("log_dir")

    train_dataset = MyDataset(args.train_dataset, wv.vocab)
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )
    val_dataset = MyDataset(args.val_dataset, wv.vocab)
    val_data_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )

    criterion = torch.nn.NLLLoss().to(DEVICE)
    optimizer = torch.optim.SGD(
        (p for p in model.parameters() if p.requires_grad),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
    )
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    total_start_time = time.time()
    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()
        train_loss, train_acc = train_func(
            model, train_data_loader, criterion, optimizer, writer
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_loss, val_acc = test_func(model, val_data_loader, criterion)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        secs = int(time.time() - start_time)
        mins = secs // 60
        secs = secs % 60
        total_secs = int(time.time() - total_start_time)
        total_mins = total_secs // 60
        total_secs = total_secs % 60
        print(
            f"Epoch {epoch} | in {mins}m {secs}s, total {total_mins}m {total_secs}s"
        )
        print(
            f"\tTrain loss: {train_loss:.3e}, Train acc: {train_acc * 100:.1f}%"
        )
        print(f"\tVal loss: {val_loss:.3e}, Val acc: {val_acc * 100:.1f}%")

        writer.add_scalar("Train/Loss", train_loss, epoch)
        writer.add_scalar("Train/Accuracy", train_acc, epoch)
        writer.add_scalar("Validation/Loss", val_loss, epoch)
        writer.add_scalar("Validation/Accuracy", val_acc, epoch)
        if epoch != EPOCHS:
            torch.save(model.state_dict(), f"{args.model_file}-{epoch}.pth")

    #plot_training(train_losses, train_accs, val_losses, val_accs)
    torch.save(model.state_dict(), f"{args.model_file}.pth")
    print("Hyperparameters:")
    print(f"\tEpochs: {EPOCHS}")
    print(f"\tBatch size: {BATCH_SIZE}")
    print(f"\tLearning rate: {LEARNING_RATE}")
    print(f"\tMomentum: {MOMENTUM}")


def train_func(model, data_loader, criterion, optimizer, writer, last_val=20):
    """
    Train the model and return the training loss and accuracy
    of the `last_val` batches.
    """
    model.train()
    losses = deque(maxlen=last_val)
    accs = deque(maxlen=last_val)
    for iteration, (labels, features) in enumerate(tqdm(data_loader)):
        labels = labels.to(DEVICE)
        features = features.to(DEVICE)

        optimizer.zero_grad()
        model.init_hidden_state()

        outputs = model(features)
        loss = criterion(outputs, labels)
        assert not torch.isnan(loss)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        accs.append((outputs.argmax(1) == labels).sum().item() / len(labels))

        if iteration % 10 == 9:
            for param_name, param_value in zip(
                model.state_dict(), model.parameters()
            ):
                if param_value.requires_grad:
                    param_name = param_name.replace(".", "/")
                    writer.add_histogram(
                        param_name,
                        param_value,
                        iteration,
                    )
                    writer.add_histogram(
                        param_name + "/grad",
                        param_value.grad,
                        iteration,
                    )
    return sum(losses) / len(losses), sum(accs) / len(accs)


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
    plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    plt.legend(["Train", "Validation"])
    plt.tight_layout()
    plt.savefig("plots/accuracy.pdf")


if __name__ == "__main__":
    main()
