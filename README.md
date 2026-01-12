MNIST-CSV-ClassicML

Classical Machine Learning pipeline for handwritten digit recognition (MNIST) using the MNIST dataset in CSV format (flattened 28x28 images). 

-----------------------------------------------------------------------
1) Project Overview
-----------------------------------------------------------------------
This project performs digit classification (0-9) using the following models:
- FastKNN (implemented from scratch)
- SVM (scikit-learn)
- Decision Tree (scikit-learn)
- Voting Ensemble (Bonus)

The workflow includes:
- Data exploration and visualization
- Normalization
- Optional PCA dimensionality reduction
- Accuracy and classification report
- Confusion matrix visualization
- Misclassified image saving and analysis

-----------------------------------------------------------------------
2) Dataset
-----------------------------------------------------------------------
Dataset used: MNIST in CSV format (dataset link :- https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)
Files:
- mnist_train.csv
- mnist_test.csv

Each row contains:
- label (digit class: 0-9)
- pixel0 ... pixel783 (grayscale pixel values in range 0-255)

Note:
The dataset already comes pre-split as training and testing files, so an additional 80/20 split was not required.

-----------------------------------------------------------------------
3) Project Workflow
-----------------------------------------------------------------------
A) Data Loading and Exploration
- Loads MNIST CSV files using Pandas
- Prints dataset shape
- Checks for missing values
- Prints class distribution
- Displays sample digits (reshaped to 28x28)

B) Preprocessing
- Normalizes pixel values from [0-255] to [0-1]
- Optional PCA to reduce dimensionality (784 -> N components)

C) Model Training
1. KNN From Scratch
- Euclidean distance
- Majority voting
- Fast vectorized distance computation

2. SVM (scikit-learn)
- RBF kernel
- Configurable hyperparameters

3. Decision Tree (scikit-learn)
- Configurable hyperparameters (depth, splits)

D) Evaluation
For each model:
- Accuracy score
- Classification report
- Confusion matrix heatmap

E) Misclassification Analysis
- Displays and saves misclassified digit images
- Saves CSV summary of misclassifications (true label vs predicted label)

F) Bonus: Voting Ensemble
- Combines KNN, SVM, and Decision Tree predictions
- Final prediction using majority voting

-----------------------------------------------------------------------
4) Results Summary (Short Report)
-----------------------------------------------------------------------
SVM achieved the highest accuracy due to strong decision boundaries in high-dimensional feature space. KNN produced competitive performance, and PCA significantly improved KNN speed by reducing dimensionality. Decision Tree trained quickly but generally underperformed compared to SVM, with overfitting tendencies on raw pixel features. Misclassification analysis showed frequent confusion between visually similar digits (e.g., 3 vs 5, 4 vs 9). A voting ensemble combining all models improved prediction stability and produced more balanced performance.

-----------------------------------------------------------------------
5) Outputs
-----------------------------------------------------------------------
All outputs are saved under:

outputs/
- confusion_matrices/
- misclassified/
- reports/
- plots/

Outputs include:
- Confusion matrix images
- Misclassified digit images
- Misclassification summary CSV files

-----------------------------------------------------------------------
6) How to Run
-----------------------------------------------------------------------
Kaggle (Recommended)
1. Upload notebook/script to Kaggle
2. Add dataset: MNIST in CSV format (mnist_train.csv and mnist_test.csv)
3. Run all cells

-----------------------------------------------------------------------
7) Requirements
-----------------------------------------------------------------------
The following packages are required:
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- tqdm
