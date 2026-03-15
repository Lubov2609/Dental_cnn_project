import pandas as pd
from torch.utils.data import DataLoader

from datasets.dental_dataset import DentalDataset
from utils.augmentations import get_train_transforms, get_val_transform

from models.resnet_model import ResNetModel
from models.custom_cnn import CustomCNN

from training.trainer import train_model

import configs.config as config


def print_dataset_info(name, dataset, loader):

    print(f"\n{name} DATASET INFO")
    print("-" * 30)

    print(f"Total images in dataset: {len(dataset)}")

    print(f"Batch size: {loader.batch_size}")

    print(f"Total batches per epoch: {len(loader)}")

    images_per_epoch = len(loader) * loader.batch_size

    print(f"Images processed per epoch (approx): {images_per_epoch}")


def run():

    # получаем список аугментаций
    train_transforms = get_train_transforms()

    val_transform = get_val_transform()

    # создаем датасеты
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

    # создаем DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE
    )

    # вывод информации о датасетах
    print("\nDATASET STATISTICS AFTER AUGMENTATION")

    print_dataset_info("TRAIN", train_dataset, train_loader)

    print_dataset_info("VALIDATION", val_dataset, val_loader)

    # создаем модели
    resnet = ResNetModel()

    custom = CustomCNN()

    print("\n==============================")
    print("Training ResNet")
    print("==============================")

    resnet_history = train_model(
        resnet,
        train_loader,
        val_loader,
        config
    )

    print("\n==============================")
    print("Training Custom CNN")
    print("==============================")

    custom_history = train_model(
        custom,
        train_loader,
        val_loader,
        config
    )

    # берем последние метрики
    resnet_metrics = resnet_history[-1]

    custom_metrics = custom_history[-1]

    # формируем таблицу сравнения
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

    print("\nCOMPARATIVE RESULTS\n")

    print(table)


if __name__ == "__main__":
    run()