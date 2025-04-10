# eda_experiments.py
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models

from config import DATASET_PATH, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, TRAINING_PORTION, NUM_CLASSES, LEARNING_RATE, NUM_EPOCHS, DROPOUT_RATE
from seed import set_seed
from data import train_transforms, val_transforms, get_dataloaders
from train_val import train_one_epoch, validate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get the datasets (to reuse for experiments like batch size)
train_loader, val_loader, train_dataset, val_dataset = get_dataloaders()

##############################
# Experiment 1: Learning Rate
##############################
learning_rates = [1e-2, 5e-3, 1e-3, 7e-4, 5e-4, 1e-4, 1e-5]
results_lr = {}

print("Starting Learning Rate Experiment")
for lr in learning_rates:
    print(f"\nTraining with learning rate: {lr}")
    set_seed(42)
    model_lr = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    for param in model_lr.parameters():
        param.requires_grad = False
    model_lr.classifier = nn.Sequential(nn.Dropout(DROPOUT_RATE),
                                        nn.Linear(model_lr.last_channel, NUM_CLASSES))
    model_lr = model_lr.to(device)
    criterion_lr = nn.CrossEntropyLoss()
    optimizer_lr = optim.Adam(model_lr.classifier.parameters(), lr=lr)
    scheduler_lr = ReduceLROnPlateau(
        optimizer_lr, mode="min", factor=0.5, patience=2)
    best_val_loss = float("inf")
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model_lr, train_loader, criterion_lr, optimizer_lr, device)
        val_loss, val_acc = validate(
            model_lr, val_loader, criterion_lr, device)
        scheduler_lr.step(val_loss)
        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    results_lr[lr] = val_acc

plt.figure(figsize=(8, 6))
plt.plot(learning_rates, list(results_lr.values()), marker='o')
plt.xscale('log')
plt.xlabel("Learning Rate")
plt.ylabel("Validation Accuracy")
plt.title("Learning Rate vs. Validation Accuracy")
plt.show()

##############################
# Experiment 2: Dropout Rate
##############################
dropout_rates = [0.1, 0.3, 0.5, 0.7, 0.9]
results_dropout = {}

print("Starting Dropout Rate Experiment")
for dr in dropout_rates:
    print(f"\nTraining with dropout rate: {dr}")
    set_seed(42)
    model_dr = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    for param in model_dr.parameters():
        param.requires_grad = False
    model_dr.classifier = nn.Sequential(nn.Dropout(dr),
                                        nn.Linear(model_dr.last_channel, NUM_CLASSES))
    model_dr = model_dr.to(device)
    criterion_dr = nn.CrossEntropyLoss()
    optimizer_dr = optim.Adam(
        model_dr.classifier.parameters(), lr=LEARNING_RATE)
    scheduler_dr = ReduceLROnPlateau(
        optimizer_dr, mode="min", factor=0.5, patience=2)
    best_val_loss = float("inf")
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model_dr, train_loader, criterion_dr, optimizer_dr, device)
        val_loss, val_acc = validate(
            model_dr, val_loader, criterion_dr, device)
        scheduler_dr.step(val_loss)
        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    results_dropout[dr] = val_acc

plt.figure(figsize=(8, 6))
plt.bar([str(dr) for dr in dropout_rates], list(results_dropout.values()))
plt.xlabel("Dropout Rate")
plt.ylabel("Validation Accuracy")
plt.title("Dropout Rate vs. Validation Accuracy")
plt.show()

##############################
# Experiment 3: Batch Size
##############################
batch_sizes = [8, 16, 32, 64, 128]
results_bs = {}

print("Starting Batch Size Experiment")
for bs in batch_sizes:
    print(f"\nTraining with batch size: {bs}")
    set_seed(42)
    train_loader_bs = DataLoader(train_dataset, batch_size=bs, shuffle=True,
                                 num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader_bs = DataLoader(val_dataset, batch_size=bs, shuffle=False,
                               num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    model_bs = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    for param in model_bs.parameters():
        param.requires_grad = False
    model_bs.classifier = nn.Sequential(nn.Dropout(DROPOUT_RATE),
                                        nn.Linear(model_bs.last_channel, NUM_CLASSES))
    model_bs = model_bs.to(device)
    criterion_bs = nn.CrossEntropyLoss()
    optimizer_bs = optim.Adam(
        model_bs.classifier.parameters(), lr=LEARNING_RATE)
    scheduler_bs = ReduceLROnPlateau(
        optimizer_bs, mode="min", factor=0.5, patience=2)
    best_val_loss = float("inf")
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model_bs, train_loader_bs, criterion_bs, optimizer_bs, device)
        val_loss, val_acc = validate(
            model_bs, val_loader_bs, criterion_bs, device)
        scheduler_bs.step(val_loss)
        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    results_bs[bs] = val_acc

plt.figure(figsize=(8, 6))
plt.plot(batch_sizes, list(results_bs.values()), marker='o')
plt.xlabel("Batch Size")
plt.ylabel("Validation Accuracy")
plt.title("Batch Size vs. Validation Accuracy")
plt.show()

##############################
# Experiment 4: Data Augmentation
##############################
transforms_dict = {
    "No_Aug": transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    "Moderate_Aug": transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    "Heavy_Aug": transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}
results_aug = {}

print("Starting Data Augmentation Experiment")
for aug_name, transform in transforms_dict.items():
    print(f"\nTraining with {aug_name} augmentation")
    set_seed(42)
    dataset_aug = datasets.ImageFolder(DATASET_PATH, transform=transform)
    train_size = int(TRAINING_PORTION * len(dataset_aug))
    val_size = len(dataset_aug) - train_size
    train_dataset_aug, val_dataset_aug = torch.utils.data.random_split(
        dataset_aug, [train_size, val_size])

    # Change transform for the validation split
    val_dataset_aug.dataset.transform = val_transforms

    train_loader_aug = DataLoader(train_dataset_aug, batch_size=BATCH_SIZE,
                                  shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader_aug = DataLoader(val_dataset_aug, batch_size=BATCH_SIZE,
                                shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    model_aug = models.mobilenet_v2(
        weights=models.MobileNet_V2_Weights.DEFAULT)
    for param in model_aug.parameters():
        param.requires_grad = False
    model_aug.classifier = nn.Sequential(nn.Dropout(DROPOUT_RATE),
                                         nn.Linear(model_aug.last_channel, NUM_CLASSES))
    model_aug = model_aug.to(device)
    criterion_aug = nn.CrossEntropyLoss()
    optimizer_aug = optim.Adam(
        model_aug.classifier.parameters(), lr=LEARNING_RATE)
    scheduler_aug = ReduceLROnPlateau(
        optimizer_aug, mode="min", factor=0.5, patience=2)
    best_val_loss = float("inf")
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model_aug, train_loader_aug, criterion_aug, optimizer_aug, device)
        val_loss, val_acc = validate(
            model_aug, val_loader_aug, criterion_aug, device)
        scheduler_aug.step(val_loss)
        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    results_aug[aug_name] = val_acc

plt.figure(figsize=(8, 6))
plt.bar(list(results_aug.keys()), list(results_aug.values()))
plt.xlabel("Augmentation Strategy")
plt.ylabel("Validation Accuracy")
plt.title("Data Augmentation vs. Validation Accuracy")
plt.show()

##############################
# Experiment 5: Freezing vs. Fine-Tuning
##############################
freezing_options = {"Freeze": True, "FineTune": False}
results_ft = {}

print("Starting Freezing vs. Fine-Tuning Experiment")
for option, freeze in freezing_options.items():
    print(f"\nTraining with {option}")
    set_seed(42)
    model_ft = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    if freeze:
        for param in model_ft.parameters():
            param.requires_grad = False
    model_ft.classifier = nn.Sequential(nn.Dropout(DROPOUT_RATE),
                                        nn.Linear(model_ft.last_channel, NUM_CLASSES))
    model_ft = model_ft.to(device)
    criterion_ft = nn.CrossEntropyLoss()
    optimizer_ft = (optim.Adam(model_ft.classifier.parameters(), lr=LEARNING_RATE)
                    if freeze else optim.Adam(model_ft.parameters(), lr=LEARNING_RATE))
    scheduler_ft = ReduceLROnPlateau(
        optimizer_ft, mode="min", factor=0.5, patience=2)
    best_val_loss = float("inf")
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model_ft, train_loader, criterion_ft, optimizer_ft, device)
        val_loss, val_acc = validate(
            model_ft, val_loader, criterion_ft, device)
        scheduler_ft.step(val_loss)
        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    results_ft[option] = val_acc

plt.figure(figsize=(8, 6))
plt.bar(list(results_ft.keys()), list(results_ft.values()))
plt.xlabel("Training Strategy")
plt.ylabel("Validation Accuracy")
plt.title("Freezing vs. Fine-Tuning Comparison")
plt.show()

##############################
# Experiment 6: Optimizer Comparison
##############################
optimizer_options = {"AdamW": optim.AdamW,
                     "Adam": optim.Adam, "SGD": optim.SGD}
results_opt = {}

print("Starting Optimizer Comparison Experiment")
for opt_name, opt_class in optimizer_options.items():
    print(f"\nTraining with {opt_name}")
    set_seed(42)
    model_opt = models.mobilenet_v2(
        weights=models.MobileNet_V2_Weights.DEFAULT)
    for param in model_opt.parameters():
        param.requires_grad = False
    model_opt.classifier = nn.Sequential(nn.Dropout(DROPOUT_RATE),
                                         nn.Linear(model_opt.last_channel, NUM_CLASSES))
    model_opt = model_opt.to(device)
    criterion_opt = nn.CrossEntropyLoss()

    if opt_name == "SGD":
        optimizer_opt = opt_class(
            model_opt.classifier.parameters(), lr=LEARNING_RATE, momentum=0.9)
    else:
        optimizer_opt = opt_class(
            model_opt.classifier.parameters(), lr=LEARNING_RATE)

    scheduler_opt = ReduceLROnPlateau(
        optimizer_opt, mode="min", factor=0.5, patience=2)
    best_val_loss = float("inf")
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model_opt, train_loader, criterion_opt, optimizer_opt, device)
        val_loss, val_acc = validate(
            model_opt, val_loader, criterion_opt, device)
        scheduler_opt.step(val_loss)
        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    results_opt[opt_name] = val_acc

plt.figure(figsize=(8, 6))
plt.bar(list(results_opt.keys()), list(results_opt.values()))
plt.xlabel("Optimizer")
plt.ylabel("Validation Accuracy")
plt.title("Optimizer Comparison: AdamW vs. Adam vs. SGD")
plt.show()
