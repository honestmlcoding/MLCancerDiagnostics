This repository demonstrates educational machine-learning workflows using a cancer-labeled dataset. It is not a medical device and must not be used for diagnosis.


pip install -r requirements.txt

python train_cancer_tabular.py --data_path data.csv --epochs 256 --batch_size 32 --hidden_dim 16 --num_layers 4

Cancer Classification (Educational ML Example)
Overview

This repository contains a machine learning example for binary classification (e.g., malignant vs. benign) using tabular data and PyTorch.

It is intended solely for educational and research purposes, such as:

Learning how to train neural networks on tabular data

Demonstrating multi-GPU (DDP) training workflows

Exploring model evaluation metrics (accuracy, precision, recall, F1, AUROC)

This project is NOT a medical device and MUST NOT be used for real-world diagnosis, treatment, or clinical decision-making.

Important Disclaimer (Read This First)

This software is provided for educational and experimental purposes only.

It is not a certified medical tool

It is not intended for clinical, diagnostic, or therapeutic use

It has not been validated for real-world medical decision-making

It should never be used to diagnose, treat, or advise patients

Any references to “cancer,” “malignant,” or “benign” are purely contextual and relate only to example datasets commonly used in machine learning education.

You assume all responsibility for how this code is used.

Features

Binary classification using a configurable MLP

Fully command-line driven (epochs, batch size, layers, etc.)

Supports single GPU or multi-GPU (2+) training

Metrics reported:

Accuracy

Precision

Recall

F1 score

AUROC

Minimal dependencies:

torch

numpy

Example Usage
2-GPU Training (Default)
python train_cancer_tabular.py --data_path data.csv --epochs 256 --batch_size 32 --hidden_dim 16 --num_layers 4

Adjustable Parameters

You can freely adjust:

--epochs

--batch_size

--hidden_dim

--num_layers

--dropout

--lr

All parameters are intended for experimentation and learning.

Dataset Notes

The script expects a CSV file with a binary label column (default: diagnosis)

Common label formats supported:

M / B

1 / 0

malignant / benign

Only numeric feature columns are used

Non-numeric columns are automatically dropped with a warning

Model Behavior & Limitations

High accuracy does not imply real-world reliability

Results depend heavily on:

Dataset quality

Train/validation split

Feature distributions

Overfitting is possible, especially on small datasets

Performance metrics are illustrative, not clinical guarantees

Intended Audience

Students learning machine learning

Engineers exploring PyTorch or DDP

Educators demonstrating ML workflows

Content creators (e.g., YouTube tutorials)
