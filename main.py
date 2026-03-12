import pandas as pd
from torch.utils.data import DataLoader

from datasets.dental_dataset import DentalDataset
from utils.augmentations import get_transforms

from models.resnet_model import ResNetModel
from models.custom_cnn import CustomCNN

from training.trainer import train_model

import configs.config as config


def run():

    train_t, val_t = get_transforms()

    train_dataset = DentalDataset(
        "data/splits/train.csv",
        "data/processed",
        train_t
    )

    val_dataset = DentalDataset(
        "data/splits/val.csv",
        "data/processed",
        val_t
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE
    )

    resnet = ResNetModel()

    custom = CustomCNN()

    print("Training ResNet")

    resnet_history = train_model(
        resnet,
        train_loader,
        val_loader,
        config
    )

    print("Training Custom CNN")

    custom_history = train_model(
        custom,
        train_loader,
        val_loader,
        config
    )

    resnet_metrics = resnet_history[-1]
    custom_metrics = custom_history[-1]

    table = pd.DataFrame({

        "Model": ["ResNet", "Custom CNN"],

        "Loss": [
            resnet_metrics["loss"],
            custom_metrics["loss"]
        ],

        "Accuracy": [
            resnet_metrics["accuracy"],
            custom_metrics["accuracy"]
        ],

        "Recall": [
            resnet_metrics["recall"],
            custom_metrics["recall"]
        ],

        "F1-score": [
            resnet_metrics["f1"],
            custom_metrics["f1"]
        ]
    })

    print("\nComparative Results\n")

    print(table)


if __name__ == "__main__":
    run()