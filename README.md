# CNN-Driven Analysis of Emotional Outbursts in Twitch Streaming Content

This repository contains code for training an image classifier using MobileNetV2 implemented in PyTorch. The project is organized into modular Python files and an accompanying Jupyter Notebook for exploration and analysis. This document provides guidance on the code organization, dataset structure, training experiments, and evaluation metrics.

## Table of Contents

- [Project Structure](#project-structure)
- [Dataset Structure](#dataset-structure)
- [Running the Code](#running-the-code)
- [Training and Validation Loss](#training-and-validation-loss)
- [Confusion Matrix](#confusion-matrix)
- [Self-Evaluation](#self-evaluation)
- [Dependencies and Setup Instructions](#dependencies-and-setup-instructions)
- [Additional Notes](#additional-notes)

## Project Structure

The repository is organized as follows:
```bash
src/
├── config.py             # Hyperparameters and settings
├── data.py               # Data preprocessing and DataLoader creation
├── eda_experiments.py    # EDA experiments (LR, dropout, batch size, etc.)
├── main.py               # Main training loop and evaluation
├── model_setup.py        # Model initialization and modifications
├── seed.py               # Random seed setup for reproducibility
├── train_val.py          # Training, validation, and utility functions
├── README.md             # Project documentation
└── dataset/              # Image dataset organized in subfolders

notebook/
├── model.ipynb           # Jupyter Notebook for model exploration

experiments/
├── batch_size.py         # Batch size experiment script
├── data_augmentation_variation.py  # Data augmentation experiments
├── dropout_rate.py       # Dropout rate experiment script
├── freezing_vs_finetuning.py  # Freezing vs. Fine-tuning experiments
├── learning_rate.py      # Learning rate experiment script
├── optimizer_comparision.py  # Optimizer comparison experiments
```


### Notebook Version

Additionally, the repository contains a Jupyter Notebook version of the complete project. The notebook includes all code cells (from data preprocessing to EDA) as an integrated document. The Python modules are provided separately for modularity and ease of debugging.

## Dataset Structure

The dataset should be organized as follows:
```bash
dataset/
├── rage/
│   ├── rage_1.jpg
│   ├── rage_2.jpg
│   ├── rage_3.jpg
│   ├── ...
│   └── rage_5400.jpg
│   
└── non-rage/
    ├── non_rage_1.jpg
    ├── non_rage_2.jpg
    ├── non_rage_3.jpg
    ├── ... 
    └── non_rage_5040.jpg
```


