import torch
from training.train import train_epoch
from training.validate import validate_epoch

def train_model(model, train_loader, val_loader, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    history = []

    for epoch in range(config.EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, acc, recall, f1 = validate_epoch(model, val_loader, device)

        history.append({
            "loss": val_loss,
            "accuracy": acc,
            "recall": recall,
            "f1": f1
        })

        print(f"Epoch {epoch+1}/{config.EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Acc: {acc:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

    return history