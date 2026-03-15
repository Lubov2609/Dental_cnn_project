import torch
from tqdm import tqdm
from utils.loss import compute_loss

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for images, labels in tqdm(loader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = compute_loss(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)