import os
import pandas as pd
from datetime import datetime


def create_log_dir():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join("logs", timestamp)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def save_history(history, model_name, log_dir):
    df = pd.DataFrame({
        "epoch": list(range(1, len(history["train_loss"]) + 1)),
        "train_loss": history["train_loss"],
        "val_loss": history["val_loss"],
        "accuracy": history["accuracy"],
        "recall": history["recall"],
        "f1": history["f1"]
    })

    path = os.path.join(log_dir, f"{model_name}_history.csv")
    df.to_csv(path, index=False)


def save_summary(table, log_dir):
    path = os.path.join(log_dir, "summary.csv")
    table.to_csv(path, index=False)