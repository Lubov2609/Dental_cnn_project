import torch
from tqdm import tqdm
from utils.loss import compute_loss
from utils.metrics import calculate_metrics

def validate_epoch(model, loader, device):
    model.eval()
    total_loss = 0
    all_outputs, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = compute_loss(outputs, labels)
            total_loss += loss.item()
            all_outputs.append([o.cpu() for o in outputs])
            all_labels.append(labels.cpu())

    # объединяем батчи
    concatenated_outputs = [torch.cat([batch[i] for batch in all_outputs], dim=0) for i in range(len(outputs))]
    concatenated_labels = torch.cat(all_labels, dim=0)

    acc, recall, f1 = calculate_metrics(concatenated_outputs, concatenated_labels)
    return total_loss / len(loader), acc, recall, f1