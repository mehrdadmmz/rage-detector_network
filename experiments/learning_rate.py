import matplotlib.pyplot as plt

learning_rates = [1e-3, 1e-4, 1e-5]
results_lr = {}

for lr in learning_rates:
    print(f"\nTraining with learning rate: {lr}")
    set_seed(42)
    # Initialize model
    model_lr = models.mobilenet_v2(pretrained=True)
    for param in model_lr.parameters():
        param.requires_grad = False
    model_lr.classifier = nn.Sequential(nn.Dropout(DROPOUT_RATE),
                                         nn.Linear(model_lr.last_channel, NUM_CLASSES))
    model_lr = model_lr.to(device)
    
    criterion_lr = nn.CrossEntropyLoss()
    optimizer_lr = optim.Adam(model_lr.classifier.parameters(), lr=lr)
    scheduler_lr = optim.lr_scheduler.ReduceLROnPlateau(optimizer_lr, mode="min", factor=0.5, patience=2)
    
    best_val_loss = float("inf")
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(model_lr, train_loader, criterion_lr, optimizer_lr, device)
        val_loss, val_acc = validate(model_lr, val_loader, criterion_lr, device)
        scheduler_lr.step(val_loss)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    results_lr[lr] = val_acc  # storing final validation accuracy

# Plotting Learning Rate vs. Validation Accuracy
plt.figure(figsize=(8,6))
plt.plot(learning_rates, list(results_lr.values()), marker='o')
plt.xscale('log')
plt.xlabel("Learning Rate")
plt.ylabel("Validation Accuracy")
plt.title("Learning Rate vs. Validation Accuracy")
plt.show()
