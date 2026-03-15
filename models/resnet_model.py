import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ResNetModel(nn.Module):
    def __init__(self, num_classes=[2,2,5,5]):
        super().__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # убираем оригинальный fc

        # отдельные fully connected слои для каждого выхода
        self.heads = nn.ModuleList([nn.Linear(in_features, n) for n in num_classes])

    def forward(self, x):
        x = self.backbone(x)
        outputs = [head(x) for head in self.heads]  # список [batch, n_classes_i]
        return outputs