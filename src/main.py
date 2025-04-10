# main.py
import time
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt

from config import NUM_EPOCHS, LEARNING_RATE, DROPOUT_RATE, NUM_CLASSES
from seed import set_seed
from data import get_dataloaders
from model_setup import get_model
from train_val import train_one_epoch, validate, get_predictions, plot_confusion_matrix

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

# Get dataloaders and datasets
train_loader, val_loader, _, _ = get_dataloaders()

# Initialize the model and move to device
model = get_model(pretrained=True)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.classifier.parameters(), lr=LEARNING_RATE)
scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)


def main():
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    for epoch in range(NUM_EPOCHS):
        start_epoch = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        epoch_time = time.time() - start_epoch

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} [{epoch_time:.1f}s]: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(),
                       "best_rage_classification_model.pth")
            print("Saved Best Model!")

    print("Training completed")
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, NUM_EPOCHS + 1), train_losses, label="Train Loss")
    plt.plot(range(1, NUM_EPOCHS + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss over Epochs")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    start = datetime.now()
    main()
    end = datetime.now()
    print(f"Time took: {end - start}.")

    # Load the best model and evaluate on the validation set
    model.load_state_dict(torch.load("best_rage_classification_model.pth"))
    preds, labels = get_predictions(model, val_loader, device)

    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(labels, preds)
    print("Confusion Matrix:")
    print(cm)

    report = classification_report(labels, preds)
    print("Classification Report:")
    print(report)

    class_names = [str(i) for i in range(NUM_CLASSES)]
    plot_confusion_matrix(cm, class_names)
