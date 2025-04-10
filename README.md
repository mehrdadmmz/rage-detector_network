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
- [Contributers](#contributers)

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

*Note:* This code uses the `datasets.ImageFolder` utility from torchvision. It assumes there are two subfolders inside the dataset and takes them as two labels. 

## Running the Code

1. **Clone the repository:**

   ```bash
   git clone https://github.com/mehrdadmmz/rage-detector_network
   cd src
   ```

2. **Install the dependencies:**

   The project relies on libraries such as NumPy, PyTorch, torchvision, tqdm, matplotlib, and scikit-learn. Install these using pip (consider using a virtual environment):
   ```bash
   pip install numpy tqdm matplotlib scikit-learn torch torchvision
   ```
4. **Prepare the dataset:**
   TODO: I have to upload it on the Huggingface and give some instructions on how to use it 
6. **Training the Model:**
   To run the main training loop, execute:
   ```bash
   python main.py
   ```
   This script trains MobileNetV2 with the specified hyperparameters, saves the best model checkpoint, and outputs evaluation results including the confusion matrix and               classification report.
8. **EDA Experiments:**
   The EDA experiments (exploring different hyperparameters such as learning rate, dropout rate, batch size, data augmentation strategies, freezing vs. fine-tuning, and optimizer     comparisons) are contained in eda_experiments.py. Run this file to see how different configurations affect model performance:
   ```bash
    python eda_experiments.py
   ```
10. **Notebook Version:**
    If you prefer working in a Jupyter Notebook environment, open the provided notebook file and run the cells interactively.

## Training and Validation Loss
During training, both the training and validation losses are recorded for every epoch. After training, a graph is plotted showing the progress of both losses over the epochs.

<img width="709" alt="image" src="https://github.com/user-attachments/assets/b2b7b327-4477-4243-a277-fe3f3cf9539c" />

## Confusion Matrix
fter training, the best model is evaluated on the validation set. A confusion matrix is generated to visualize the classification performance, along with a detailed classification report.

<img width="476" alt="image" src="https://github.com/user-attachments/assets/e09fc7b9-ee64-47d5-8b39-4ea863f1628f" />

## Self-Evaluation
## Dependencies and Setup Instructions
- Python 3.6 or later:
  - The code has been tested with Python 3.8+.
- PyTorch and Torchvision:
  - Install these libraries using pip:
  ```bash
    pip install torch torchvision
  ```
- Other Libraries:
  - The project requires:
      - numpy
      - tqdm
      - matplotlib
      - scikit-learn
  - Install them with:
  ```bash
  pip install numpy tqdm matplotlib scikit-learn
  ```
- No Special Software Requirements:
  - There are no dependencies on external tools such as Unity.
## Additional Notes
- Reproducibility:
  - The code sets a fixed seed (default 42) in seed.py to ensure reproducibility. You can modify the seed as needed.
- Modularity:
  - The modular organization allows easy experimentation with different components (e.g., model architecture, data augmentation, hyperparameters).
- Future Work:
  - Future improvements might include adding more model architectures, further hyperparameter tuning, or an advanced data augmentation strategy.
 
## Contributors
- [Mehrdad Momeni Zadeh](https://github.com/mhrddmmz) – mma236@sfu.ca
- [Zheng (Arthur) Li](https://github.com/Mercury-AL) – zla229@sfu.ca
- [Daniel Surina](https://github.com/sosokokos) – dsa108@sfu.ca


