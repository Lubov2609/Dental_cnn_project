import torchvision.transforms as transforms


def get_train_transforms():

    base = transforms.Compose([
        transforms.Resize((224, 448)),
        transforms.ToTensor()
    ])

    flip = transforms.Compose([
        transforms.Resize((224, 448)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor()
    ])

    rotate = transforms.Compose([
        transforms.Resize((224, 448)),
        transforms.RandomRotation((5,5)),
        transforms.ToTensor()
    ])

    brighten = transforms.Compose([
        transforms.Resize((224, 448)),
        transforms.ColorJitter(brightness=0.2),
        transforms.ToTensor()
    ])

    darken = transforms.Compose([
        transforms.Resize((224, 448)),
        transforms.ColorJitter(brightness=0.5),
        transforms.ToTensor()
    ])

    return [base, flip, rotate, brighten, darken]


def get_val_transform():

    return transforms.Compose([
        transforms.Resize((224,448)),
        transforms.ToTensor()
    ])