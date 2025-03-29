optimizer_options = {"Adam": optim.Adam, "SGD": optim.SGD}
results_opt = {}

for opt_name, opt_class in optimizer_options.items():
    print(f"\nTraining with {opt_name}")
    set_seed(42)
    model_opt = models.mobilenet_v2(pretrained=True)
    for param in model_opt.parameters():
        param.requires_grad = False
    model_opt.classifier = nn.Sequential(nn.Dropout(DROPOUT_RATE),
                                          nn.Linear(model_opt.last_channel, NUM_CLASSES))
    model_opt = model_opt.to(device)
    
    criterion_opt = nn.CrossEntropyLoss()
    if opt_name == "Adam":
        optimizer_opt = opt_class(model_opt.classifier.parameters(), lr=LEARNING_RATE)
    else:
        optimizer_opt = opt_class(model_opt.classifier.parameters(), lr=LEARNING_RATE, momentum=0.9)
    scheduler_opt = optim.lr_scheduler.ReduceLROnPlateau(optimizer_opt, mode="min", factor=0.5, patience=2)
    
    best_val_loss = float("inf")
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(model_opt, train_loader, criterion_opt, optimizer_opt, device)
        val_loss, val_acc = validate(model_opt, val_loader, criterion_opt, device)
        scheduler_opt.step(val_loss)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    results_opt[opt_name] = val_acc

# Plotting Optimizer Comparison
plt.figure(figsize=(8,6))
plt.bar(list(results_opt.keys()), list(results_opt.values()))
plt.xlabel("Optimizer")
plt.ylabel("Validation Accuracy")
plt.title("Optimizer Comparison: Adam vs. SGD")
plt.show()
