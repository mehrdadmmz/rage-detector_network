# Define different transform pipelines
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
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

results_aug = {}

for aug_name, transform in transforms_dict.items():
    print(f"\nTraining with {aug_name} augmentation")
    set_seed(42)
    # Load dataset with new training transform
    dataset_aug = datasets.ImageFolder(DATASET_PATH, transform=transform)
    train_size = int(TRAINING_PORTION * len(dataset_aug))
    val_size = len(dataset_aug) - train_size
    train_dataset_aug, val_dataset_aug = random_split(dataset_aug, [train_size, val_size])
    # Use a standard transform for validation
    val_dataset_aug.dataset.transform = val_transforms
    
    train_loader_aug = DataLoader(train_dataset_aug, batch_size=BATCH_SIZE,
                                  shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader_aug = DataLoader(val_dataset_aug, batch_size=BATCH_SIZE,
                                shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    
    model_aug = models.mobilenet_v2(pretrained=True)
    for param in model_aug.parameters():
        param.requires_grad = False
    model_aug.classifier = nn.Sequential(nn.Dropout(DROPOUT_RATE),
                                          nn.Linear(model_aug.last_channel, NUM_CLASSES))
    model_aug = model_aug.to(device)
    
    criterion_aug = nn.CrossEntropyLoss()
    optimizer_aug = optim.Adam(model_aug.classifier.parameters(), lr=LEARNING_RATE)
    scheduler_aug = optim.lr_scheduler.ReduceLROnPlateau(optimizer_aug, mode="min", factor=0.5, patience=2)
    
    best_val_loss = float("inf")
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(model_aug, train_loader_aug, criterion_aug, optimizer_aug, device)
        val_loss, val_acc = validate(model_aug, val_loader_aug, criterion_aug, device)
        scheduler_aug.step(val_loss)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    results_aug[aug_name] = val_acc

# Plotting Augmentation Type vs. Validation Accuracy
plt.figure(figsize=(8,6))
plt.bar(list(results_aug.keys()), list(results_aug.values()))
plt.xlabel("Augmentation Strategy")
plt.ylabel("Validation Accuracy")
plt.title("Data Augmentation vs. Validation Accuracy")
plt.show()
