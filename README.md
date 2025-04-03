# ELE489_HW1
 Implementation of the k-NN algorithm and its test on the Wine Dataset from the UCI  Machine Learning Repository.
# k-Nearest Neighbors (k-NN) Implementation

This repository contains an implementation of the k-Nearest Neighbors (k-NN) algorithm from scratch and its application to the Wine Dataset from the UCI Machine Learning Repository.

## Project Overview
This project is part of the **ELE 489: Fundamentals of Machine Learning** course. The goal is to implement k-NN without using `sklearn.neighbors.KNeighborsClassifier`, explore different values of *k*, and compare distance metrics such as Euclidean and Manhattan distances.

## Files in This Repository
- **`knn.py`**: Implements the k-NN algorithm, including training, classification, and accuracy evaluation.
- **`analysis.ipynb`**: A Jupyter Notebook that:
  - Loads and visualizes the dataset.
  - Performs preprocessing (e.g., normalization, splitting data into train/test sets).
  - Trains and evaluates k-NN for different values of *k* and distance metrics.
  - Plots accuracy vs. *k* and displays the confusion matrix and classification report.
- **`README.md`**: Instructions for running the code and understanding the project.

## Installation and Setup

- Install the required dependencies:
   ```bash
   pip install pandas
   pip install numpy
   pip install matplotlib
   pip install seaborn
   pip install scikit-learn
   pip install jupyter
   ```
- Run the Jupyter Notebook:
   ```bash
   jupyter notebook analysis.ipynb
   ```

## Usage
- The `knn.py` script contains the `SimpleKNN` class, which can be imported and used to train and classify new data.
- The Jupyter Notebook walks through data visualization, training, and evaluation.

## Results
- The notebook provides accuracy scores for different *k* values.
- A confusion matrix and classification report are generated to evaluate model performance.

https://deepnote.com/app/ndungu/Implementing-KNN-Algorithm-on-the-Iris-Dataset-e7c16493-500c4248-be54-9389de603f16 
https://www.kaggle.com/code/prashant111/knn-classifier-tutorial 
https://www.w3schools.com/python/python_ml_confusion_matrix.asp)
