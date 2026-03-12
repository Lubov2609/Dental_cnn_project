import torchvision.transforms as transforms


def get_transforms():

    train_transform = transforms.Compose([

        # transforms.Resize((768, 1536)),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),

        transforms.RandomRotation(5),

        transforms.ColorJitter(brightness=0.1),

        transforms.ToTensor()
    ])

    val_transform = transforms.Compose([

        transforms.Resize((768, 1536)),

        transforms.ToTensor()
    ])

    return train_transform, val_transform