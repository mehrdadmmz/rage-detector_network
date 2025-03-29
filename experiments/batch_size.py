batch_sizes = [16, 32, 64]
results_bs = {}

for bs in batch_sizes:
    print(f"\nTraining with batch size: {bs}")
    set_seed(42)
    # Reinitialize DataLoaders with new batch size
    train_loader_bs = DataLoader(train_dataset, batch_size=bs, shuffle=True,
                                 num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader_bs = DataLoader(val_dataset, batch_size=bs, shuffle=False,
                               num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    
    model_bs = models.mobilenet_v2(pretrained=True)
    for param in model_bs.parameters():
        param.requires_grad = False
    model_bs.classifier = nn.Sequential(nn.Dropout(DROPOUT_RATE),
                                         nn.Linear(model_bs.last_channel, NUM_CLASSES))
    model_bs = model_bs.to(device)
    
    criterion_bs = nn.CrossEntropyLoss()
    optimizer_bs = optim.Adam(model_bs.classifier.parameters(), lr=LEARNING_RATE)
    scheduler_bs = optim.lr_scheduler.ReduceLROnPlateau(optimizer_bs, mode="min", factor=0.5, patience=2)
    
    best_val_loss = float("inf")
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(model_bs, train_loader_bs, criterion_bs, optimizer_bs, device)
        val_loss, val_acc = validate(model_bs, val_loader_bs, criterion_bs, device)
        scheduler_bs.step(val_loss)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    results_bs[bs] = val_acc

# Plotting Batch Size vs. Validation Accuracy
plt.figure(figsize=(8,6))
plt.plot(batch_sizes, list(results_bs.values()), marker='o')
plt.xlabel("Batch Size")
plt.ylabel("Validation Accuracy")
plt.title("Batch Size vs. Validation Accuracy")
plt.show()
