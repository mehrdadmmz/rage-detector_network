# data_loader.py
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from config import DATASET_PATH, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY


def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])


def get_dataloaders(train_dataset, val_dataset):
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS,
                              pin_memory=PIN_MEMORY)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=NUM_WORKERS,
                            pin_memory=PIN_MEMORY)

    return train_loader, val_loader


def load_datasets(train=True, val=True):
    train_dataset = datasets.ImageFolder(DATASET_PATH,
                                         transform=get_transforms(train=True))
    val_dataset = datasets.ImageFolder(DATASET_PATH,
                                       transform=get_transforms(train=False))

    return train_dataset, val_dataset
