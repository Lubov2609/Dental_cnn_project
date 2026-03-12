import torch
import pandas as pd

from utils.metrics import calculate_metrics
from training.train import train_epoch
from training.validate import validate


def train_model(model, train_loader, val_loader, config):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LR
    )

    criterion = torch.nn.BCELoss()

    history = []

    for epoch in range(config.EPOCHS):

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device
        )

        val_loss, preds, trues = validate(
            model,
            val_loader,
            criterion,
            device
        )

        acc, recall, f1 = calculate_metrics(
            trues,
            preds
        )

        history.append({
            "loss": val_loss,
            "accuracy": acc,
            "recall": recall,
            "f1": f1
        })

        print(
            f"Epoch {epoch+1} | Loss {val_loss:.4f} | Acc {acc:.4f}"
        )

    return history