freezing_options = {"Freeze": True, "FineTune": False}
results_ft = {}

for option, freeze in freezing_options.items():
    print(f"\nTraining with {option}")
    set_seed(42)
    model_ft = models.mobilenet_v2(pretrained=True)
    if freeze:
        for param in model_ft.parameters():
            param.requires_grad = False
    # Replace the classifier in both cases
    model_ft.classifier = nn.Sequential(nn.Dropout(DROPOUT_RATE),
                                         nn.Linear(model_ft.last_channel, NUM_CLASSES))
    model_ft = model_ft.to(device)
    
    criterion_ft = nn.CrossEntropyLoss()
    if freeze:
        optimizer_ft = optim.Adam(model_ft.classifier.parameters(), lr=LEARNING_RATE)
    else:
        optimizer_ft = optim.Adam(model_ft.parameters(), lr=LEARNING_RATE)
    scheduler_ft = optim.lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode="min", factor=0.5, patience=2)
    
    best_val_loss = float("inf")
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(model_ft, train_loader, criterion_ft, optimizer_ft, device)
        val_loss, val_acc = validate(model_ft, val_loader, criterion_ft, device)
        scheduler_ft.step(val_loss)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    results_ft[option] = val_acc

# Plotting Freezing vs. Fine-Tuning
plt.figure(figsize=(8,6))
plt.bar(list(results_ft.keys()), list(results_ft.values()))
plt.xlabel("Training Strategy")
plt.ylabel("Validation Accuracy")
plt.title("Freezing vs. Fine-Tuning Comparison")
plt.show()
