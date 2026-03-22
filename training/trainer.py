import torch
from training.train import train_epoch
from training.validate import validate_epoch

def train_model(model, train_loader, val_loader, config, log_path=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)

    history = {
        "train_loss": [],
        "val_loss": [],
        "accuracy": [],
        "recall": [],
        "f1": []
    }

    log_file = None

    if log_path:
        log_file = open(log_path, "w")
        log_file.write("epoch,train_loss,val_loss,accuracy,recall,f1\n")

    for epoch in range(config.EPOCHS):

        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, acc, recall, f1 = validate_epoch(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["accuracy"].append(acc)
        history["recall"].append(recall)
        history["f1"].append(f1)

        log_line = f"{epoch+1},{train_loss},{val_loss},{acc},{recall},{f1}\n"

        print(
            f"Epoch {epoch+1}/{config.EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Acc: {acc:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}"
        )

        if log_file:
            log_file.write(log_line)

    if log_file:
        log_file.close()

    return history