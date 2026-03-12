import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class ResNetModel(nn.Module):

    def __init__(self):

        super().__init__()

        # загрузка предобученных весов
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)

        in_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Sequential(

            nn.Linear(in_features, 128),

            nn.ReLU(),

            nn.Linear(128, 4),

            nn.Sigmoid()
        )

    def forward(self, x):

        return self.backbone(x)