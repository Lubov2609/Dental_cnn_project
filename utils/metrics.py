import torch
from sklearn.metrics import accuracy_score, recall_score, f1_score

def calculate_metrics(outputs, targets):
    """
    outputs: list of tensors [batch, n_classes_i]
    targets: tensor [batch, n_outputs]
    """
    preds = [torch.argmax(o, dim=1).cpu().numpy() for o in outputs]
    targets_np = targets.cpu().numpy()

    acc_list, recall_list, f1_list = [], [], []

    for i in range(len(outputs)):
        acc_list.append(accuracy_score(targets_np[:, i], preds[i]))
        recall_list.append(recall_score(targets_np[:, i], preds[i], average='macro'))
        f1_list.append(f1_score(targets_np[:, i], preds[i], average='macro'))

    return sum(acc_list)/len(acc_list), sum(recall_list)/len(recall_list), sum(f1_list)/len(f1_list)