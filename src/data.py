# data.py
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
from config import DATASET_PATH, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, TRAINING_PORTION

# Define transforms for training and validation
train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


def get_dataloaders():
    # Create the dataset
    dataset = datasets.ImageFolder(DATASET_PATH, transform=train_transforms)

    # Split into training and validation portions
    train_size = int(TRAINING_PORTION * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Use a different transform for validation
    val_dataset.transform = val_transforms

    # Create loaders
    train_loader = DataLoader(train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              num_workers=NUM_WORKERS,
                              pin_memory=PIN_MEMORY)
    val_loader = DataLoader(val_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=False,
                            num_workers=NUM_WORKERS,
                            pin_memory=PIN_MEMORY)

    return train_loader, val_loader, train_dataset, val_dataset
