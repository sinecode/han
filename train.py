import argparse
import time
from datetime import datetime
from pathlib import Path
from collections import deque

import torch
from tensorboardX import SummaryWriter
from gensim.models import KeyedVectors
import pandas as pd
from tqdm import tqdm

from dataset import SentWordDataset, WordDataset
from models import Han, Wan
from test import test_func
from config import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    MOMENTUM,
    PATIENCE,
    PADDING,
    DEVICE,
    TQDM,
    WORD_HIDDEN_SIZE,
    SENT_HIDDEN_SIZE,
    MODEL_DIR,
    LOG_DIR,
    Yelp,
    Yahoo,
    Amazon,
    Synthetic,
)


def main():
    parser = argparse.ArgumentParser(
        description="Train the WAN or the HAN model"
    )
    parser.add_argument(
        "dataset",
        choices=["yelp", "yahoo", "amazon", "synthetic"],
        help="Choose the dataset",
    )
    parser.add_argument(
        "model", choices=["wan", "han"], help="Choose the model to be trained",
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

    train_df = pd.read_csv(dataset_config.TRAIN_DATASET).fillna("")
    train_documents = train_df.text
    train_labels = train_df.label
    if args.model == "wan":
        train_dataset = WordDataset(
            train_documents,
            train_labels,
            wv.vocab,
            dataset_config.WORDS_PER_DOC[PADDING],
        )
    else:
        train_dataset = SentWordDataset(
            train_documents,
            train_labels,
            wv.vocab,
            dataset_config.SENT_PER_DOC[PADDING],
            dataset_config.WORDS_PER_SENT[PADDING],
        )
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )

    val_df = pd.read_csv(dataset_config.VAL_DATASET).fillna("")
    val_documents = val_df.text
    val_labels = val_df.label
    if args.model == "wan":
        val_dataset = WordDataset(
            val_documents,
            val_labels,
            wv.vocab,
            dataset_config.WORDS_PER_DOC[PADDING],
        )
    else:
        val_dataset = SentWordDataset(
            val_documents,
            val_labels,
            wv.vocab,
            dataset_config.SENT_PER_DOC[PADDING],
            dataset_config.WORDS_PER_SENT[PADDING],
        )
    val_data_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=True
    )

    logdir = Path(f"{LOG_DIR}/{args.dataset}/{args.model}")
    logdir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(
        str(logdir / datetime.now().strftime("%Y%m%d-%H%M%S"))
    )

    if args.model == "wan":
        model = Wan(
            embedding_matrix=wv.vectors,
            word_hidden_size=WORD_HIDDEN_SIZE,
            num_classes=len(train_labels.unique()),
            batch_size=BATCH_SIZE,
        ).to(DEVICE)
    else:
        model = Han(
            embedding_matrix=wv.vectors,
            word_hidden_size=WORD_HIDDEN_SIZE,
            sent_hidden_size=SENT_HIDDEN_SIZE,
            num_classes=len(train_labels.unique()),
            batch_size=BATCH_SIZE,
        ).to(DEVICE)

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
    best_val_loss = 100_000
    actual_patience = 0
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
            f"\tTrain loss: {train_loss:.4}, Train acc: {train_acc * 100:.1f}%"
        )
        print(f"\tVal loss: {val_loss:.4}, Val acc: {val_acc * 100:.1f}%")

        writer.add_scalar("Train/Loss", train_loss, epoch)
        writer.add_scalar("Train/Accuracy", train_acc, epoch)
        writer.add_scalar("Validation/Loss", val_loss, epoch)
        writer.add_scalar("Validation/Accuracy", val_acc, epoch)

        # Early stopping with patience
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        else:
            actual_patience += 1
            if actual_patience == PATIENCE:
                break

    writer.add_text(
        "Hyperparameters",
        f"BATCH_SIZE = {BATCH_SIZE}; "
        f"LEARNING_RATE = {LEARNING_RATE}; "
        f"MOMENTUM = {MOMENTUM}; "
        f"PATIENCE = {PATIENCE}; ",
        f"PADDING = {PADDING}",
    )
    writer.close()

    modeldir = Path(MODEL_DIR)
    modeldir.mkdir(parents=True, exist_ok=True)
    torch.save(
        model.state_dict(),
        f"{modeldir}/{args.dataset}-{args.model}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.pth",
    )


def train_func(model, data_loader, criterion, optimizer, writer, last_val=20):
    """
    Train the model and return the training loss and accuracy
    of the `last_val` batches.
    """
    model.train()
    losses = deque(maxlen=last_val)
    accs = deque(maxlen=last_val)
    for iteration, (labels, features) in tqdm(
        enumerate(data_loader), total=len(data_loader), disable=(not TQDM)
    ):
        labels = labels.to(DEVICE)
        features = features.to(DEVICE)

        batch_size = len(labels)
        optimizer.zero_grad()
        model.init_hidden_state(batch_size)

        predictions = model(features)
        loss = criterion(predictions, labels)
        assert not torch.isnan(loss)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        accs.append(
            (predictions.argmax(1) == labels).sum().item() / batch_size
        )

        if iteration % 1_000 == 999:
            for param_name, param_value in zip(
                model.state_dict(), model.parameters()
            ):
                if param_value.requires_grad:
                    param_name = param_name.replace(".", "/")
                    writer.add_histogram(
                        param_name, param_value, iteration,
                    )
                    # writer.add_histogram(
                    #    param_name + "/grad", param_value.grad, iteration,
                    # )
    return sum(losses) / len(losses), sum(accs) / len(accs)


if __name__ == "__main__":
    main()
