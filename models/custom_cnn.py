import torch.nn as nn


class CustomCNN(nn.Module):

    def __init__(self):

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

        self.fc = nn.Sequential(

            nn.Linear(128 * 48 * 96, 128),

            nn.ReLU(),

            nn.Linear(128, 4),

            nn.Softmax(dim=1)
        )

    def forward(self, x):

        x = self.features(x)

        x = self.flatten(x)

        x = self.fc(x)

        return x