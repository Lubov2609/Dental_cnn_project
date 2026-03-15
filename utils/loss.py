import torch
import torch.nn as nn

def compute_loss(outputs, targets):
    """
    outputs: list of [batch, n_classes_i]
    targets: tensor [batch, n_outputs]
    """
    loss = 0
    for i, out in enumerate(outputs):
        loss += nn.CrossEntropyLoss()(out, targets[:, i])
    return loss