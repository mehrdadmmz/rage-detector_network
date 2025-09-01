# CNN-Driven Analysis of Emotional Outbursts in Twitch Streaming Content

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?logo=pytorch)
![Torchvision](https://img.shields.io/badge/Lib-Torchvision-orange)
![NumPy](https://img.shields.io/badge/Lib-NumPy-013243?logo=numpy)
![Matplotlib](https://img.shields.io/badge/Lib-Matplotlib-11557c?logo=plotly)
![Scikit-learn](https://img.shields.io/badge/ML-Scikit--learn-F7931E?logo=scikitlearn)
![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-F37626?logo=jupyter)

This repository contains code for training an image classifier using MobileNetV2 implemented in PyTorch. The project is organized into modular Python files and an accompanying Jupyter Notebook for exploration and analysis. This document provides guidance on the code organization, dataset structure, training experiments, and evaluation metrics.

## Table of Contents

- [Project Structure](#project-structure)
- [Dataset Structure](#dataset-structure)
- [Pre Trained Model Structure](#pre-trained-model-structure)
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
└── README.md             # Project documentation

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

## Pre Trained Model Structure

<img width="810" alt="image" src="https://github.com/user-attachments/assets/d42b3e7f-158b-46b4-87a6-1a6036e7237f" />
<img width="719" alt="image" src="https://github.com/user-attachments/assets/717d84a1-e33e-4846-a679-3cac7cb870cc" />

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
   TODO: I will upload it on the Huggingface and give some instructions on how to use it 
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
During training, both the training and validation losses are recorded for every epoch. After training, a graph is plotted showing the progress of both losses over the epochs. Our trained model shows a steady decrease in both training and validation losses through the epochs, which signals incremental learning with less significant overfitting. This also appears in the general accuracy of 86%, which displays a consistent capacity to distinguish between rage and non-rage states.


<img width="709" alt="image" src="https://github.com/user-attachments/assets/b2b7b327-4477-4243-a277-fe3f3cf9539c" />

## Confusion Matrix
After training, the best model is evaluated on the validation set. A confusion matrix is generated to visualize the classification performance, along with a detailed classification report. The confusion matrix illustrates that most frames are well-labeled, with 964 of 1,075 "rage" frames and 885 of 1,085 "non-rage" frames being predicted correctly. Although there are a few misclassifications (111 rage frames misclassified as non-rage and 200 non-rage frames misclassified as rage), the classification report shows strong precision, recall, and F1-scores (0.82-0.90). In particular, we see slightly higher recall for rage (0.90) than non-rage (0.82), indicating that the model more consistently identifies rage frames.
Overall, the above findings demonstrate the model's competence in detecting rage-related cues with balanced precision and recall for both classes. This performance offers a solid foundation for practical applications requiring precise emotion or state detection. Nevertheless, further fine-tuning and expansion of the training set may improve its stability and generalizability in real-world situations.


<img width="476" alt="image" src="https://github.com/user-attachments/assets/e09fc7b9-ee64-47d5-8b39-4ea863f1628f" />

## Self-Evaluation

Overall, the project closely followed our initial proposal. We successfully gathered the data, trained and fine-tuned the model, and performed thorough exploratory data analysis to understand the underlying patterns. While the core plan remained intact, we refined some methods along the way to optimize performance and address unforeseen challenges. All special dependencies are documented in the README. This project not only met but, in some aspects, exceeded our original goals, resulting in a robust and efficient emotion classification system.

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
- [Mehrdad Momeni zadeh](https://github.com/mhrddmmz) – mma236@sfu.ca
- [Zheng (Arthur) Li](https://github.com/Mercury-AL) – zla229@sfu.ca
- [Daniel Surina](https://github.com/sosokokos) – dsa108@sfu.ca


