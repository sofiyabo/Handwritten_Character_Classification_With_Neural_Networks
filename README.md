# Handwritten Character Classification with Neural Networks

Practical work developed for the Artificial Intelligence Engineering program at Universidad de San Andrés. This project focuses on building, training, and optimizing Multi-Layer Perceptron (MLP) models to classify handwritten characters from the EMNIST dataset.

---

## Overview

This notebook covers the full deep learning pipeline for a multiclass classification problem:

1. **Data Preprocessing:** loading image data, normalization (pixel values scaled to [0, 1]), and train/validation/test split (70/15/15).
2. **Baseline Model:** MLP with architecture [784 → 128 → 64 → 49], trained and evaluated on 49 character classes.
3. **Advanced Optimization:** implementing and comparing multiple training improvements:
   - Learning rate scheduling (linear and exponential decay)
   - Mini-batch SGD
   - Adam optimizer
   - L2 regularization
   - Early stopping
4. **Evaluation:** comparing models using accuracy, cross-entropy loss, F1-score (macro), and confusion matrices.

---

## Key Concepts

- Multi-Layer Perceptron (MLP) architecture
- Multiclass classification (49 classes)
- Backpropagation and gradient descent
- Optimization algorithms: SGD, Adam
- Regularization: L2 weight decay, early stopping
- Learning rate scheduling
- Model evaluation: accuracy, F1-score (macro), cross-entropy

---

## Project Structure

```
├── data/
│   ├── raw/                # X_images.npy, Y_images.npy
├── src/
│   ├── models.py           # MLP implementation
│   ├── preprocessing.py    # Data loading and normalization
│   ├── data_splitting.py   # Train/val/test split
│   ├── metrics.py          # Accuracy, F1, cross-entropy
│   └── utils.py            # Helper functions
├── notebooks
│   ├──Entrega_TP3.ipynb       # Main notebook
└── README.md
```
