import torchvision.transforms as transforms


def get_train_transforms():

    base = transforms.Compose([
        transforms.Resize((224, 448)),
        transforms.Grayscale(num_output_channels=1),  # 👈 ключевое
        transforms.ToTensor()
    ])

    flip = transforms.Compose([
        transforms.Resize((224, 448)),
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor()
    ])

    rotate = transforms.Compose([
        transforms.Resize((224, 448)),
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomRotation((5, 5)),
        transforms.ToTensor()
    ])

    # ⚠️ ColorJitter для grayscale — ограниченно полезен
    brighten = transforms.Compose([
        transforms.Resize((224, 448)),
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.2)
        ], p=1.0),
        transforms.ToTensor()
    ])

    darken = transforms.Compose([
        transforms.Resize((224, 448)),
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.5)
        ], p=1.0),
        transforms.ToTensor()
    ])

    return [base, flip, rotate, brighten, darken]


def get_val_transform():

    return transforms.Compose([
        transforms.Resize((224, 448)),
        transforms.Grayscale(num_output_channels=1),  # 👈 обязательно
        transforms.ToTensor()
    ])