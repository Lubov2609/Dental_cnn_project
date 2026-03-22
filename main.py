import pandas as pd
from torch.utils.data import DataLoader

from datasets.dental_dataset import DentalDataset
from utils.augmentations import get_train_transforms, get_val_transform
from utils.plots import plot_training
from utils.logger import create_log_dir, save_history, save_summary

from models.resnet_model import ResNetModel
from models.custom_cnn import CustomCNN

from training.trainer import train_model

import configs.config as config


def run():

    train_transforms = get_train_transforms()
    val_transform = get_val_transform()
    log_dir = create_log_dir()

    train_dataset = DentalDataset(
        "data/splits/train.csv",
        "data/processed",
        train_transforms
    )

    val_dataset = DentalDataset(
        "data/splits/val.csv",
        "data/processed",
        [val_transform]
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

    print("\nTraining ResNet\n")

    resnet_history = train_model(
        resnet,
        train_loader,
        val_loader,
        config,
        log_path=f"{log_dir}/resnet_log.csv"
    )

    print("\nTraining Custom CNN\n")

    custom_history = train_model(
        custom,
        train_loader,
        val_loader,
        config,
        log_path=f"{log_dir}/custom_log.csv"
    )

    resnet_metrics = {
        "loss": resnet_history["val_loss"][-1],
        "accuracy": resnet_history["accuracy"][-1],
        "recall": resnet_history["recall"][-1],
        "f1": resnet_history["f1"][-1]
    }

    custom_metrics = {
        "loss": custom_history["val_loss"][-1],
        "accuracy": custom_history["accuracy"][-1],
        "recall": custom_history["recall"][-1],
        "f1": custom_history["f1"][-1]
    }

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
    plot_training(resnet_history, "ResNet")
    plot_training(custom_history, "Custom CNN")
    save_history(resnet_history, "resnet", log_dir)
    save_history(custom_history, "custom", log_dir)
    save_summary(table, log_dir)
if __name__ == "__main__":
    run()