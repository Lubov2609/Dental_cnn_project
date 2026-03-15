import torch
import torch.nn as nn

class CustomCNN(nn.Module):
    def __init__(self, num_classes=[2,2,5,5]):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.flatten = nn.Flatten()
        self.num_classes = num_classes
        self.heads = nn.ModuleList()  # список "голов" для каждой задачи

        # временный слой для вычисления размера
        self._initialize_heads()

    def _initialize_heads(self):
        # создаем dummy input для вычисления размера flatten
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 448)  # твой размер картинок
            x = self.features(dummy_input)
            in_features = x.numel()  # размер после flatten
        self.in_features = in_features
        # создаем головы
        for n in self.num_classes:
            self.heads.append(nn.Sequential(
                nn.Linear(self.in_features, 128),
                nn.ReLU(),
                nn.Linear(128, n)
            ))

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        outputs = [head(x) for head in self.heads]
        return outputs