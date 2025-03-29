dropout_rates = [0.3, 0.5, 0.7]
results_dropout = {}

for dr in dropout_rates:
    print(f"\nTraining with dropout rate: {dr}")
    set_seed(42)
    model_dr = models.mobilenet_v2(pretrained=True)
    for param in model_dr.parameters():
        param.requires_grad = False
    model_dr.classifier = nn.Sequential(nn.Dropout(dr),
                                         nn.Linear(model_dr.last_channel, NUM_CLASSES))
    model_dr = model_dr.to(device)
    
    criterion_dr = nn.CrossEntropyLoss()
    optimizer_dr = optim.Adam(model_dr.classifier.parameters(), lr=LEARNING_RATE)
    scheduler_dr = optim.lr_scheduler.ReduceLROnPlateau(optimizer_dr, mode="min", factor=0.5, patience=2)
    
    best_val_loss = float("inf")
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(model_dr, train_loader, criterion_dr, optimizer_dr, device)
        val_loss, val_acc = validate(model_dr, val_loader, criterion_dr, device)
        scheduler_dr.step(val_loss)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    results_dropout[dr] = val_acc

# Plotting Dropout Rate vs. Validation Accuracy
plt.figure(figsize=(8,6))
plt.bar([str(dr) for dr in dropout_rates], list(results_dropout.values()))
plt.xlabel("Dropout Rate")
plt.ylabel("Validation Accuracy")
plt.title("Dropout Rate vs. Validation Accuracy")
plt.show()
