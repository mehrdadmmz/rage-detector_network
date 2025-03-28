# train.py
import time
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from model import get_model
from config import NUM_EPOCHS, LEARNING_RATE
from utils import set_seed
from data_loader import load_datasets, get_dataloaders

# Set the seed
set_seed()


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for inputs, labels in tqdm(loader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()  # reset the gradients to 0

        # forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        # get the predicted class by finding the index with the max output val
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    # disable gradient computation for the validation
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Validation", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            # forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model().to(device)

    train_dataset, val_dataset = load_datasets()
    train_loader, val_loader = get_dataloaders(train_dataset, val_dataset)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                     factor=0.5, patience=2)
    best_val_loss = float("inf")

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader,
                                                criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        epoch_time = time.time() - start_time

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} [{epoch_time:.1f}s]: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(),
                       "best_rage_classification_model.pth")
            print("Saved Best Model!")
    print("Training completed")


if __name__ == "__main__":
    main()
