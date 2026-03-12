import torch


def validate(model, loader, criterion, device):

    model.eval()

    total_loss = 0

    preds = []
    trues = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            total_loss += loss.item()

            preds.extend(outputs.cpu().numpy())
            trues.extend(labels.cpu().numpy())

    return total_loss / len(loader), preds, trues