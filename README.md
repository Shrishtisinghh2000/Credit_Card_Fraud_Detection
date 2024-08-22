# Credit_Card_Fraud_Detection


This project aims to detect fraudulent credit card transactions using machine learning techniques. The dataset used for this project is highly imbalanced, with a very small percentage of fraudulent transactions. The project demonstrates the use of undersampling, oversampling with SMOTE, and Logistic Regression to address the imbalance and build an effective predictive model.


## Table of Contents


Project Overview

Dataset

Project Structure

Dependencies

Usage

Model Description

Results

Acknowledgements

## Project Overview


Credit card fraud detection is a critical application of machine learning. This project focuses on detecting fraudulent transactions by implementing various data preprocessing techniques, feature scaling, and using Logistic Regression as the predictive model. The project also addresses class imbalance, a common challenge in fraud detection, by employing undersampling and SMOTE (Synthetic Minority Over-sampling Technique).


## Dataset


The dataset used in this project is a publicly available dataset containing transactions made by European cardholders over two days in September 2013. It includes a total of 284,807 transactions, out of which 492 are fraudulent.


Features:

30 numerical input variables (V1 to V28, Amount, and Time).


Target: Class (0 for legal transactions, 1 for fraudulent transactions).


## Dependencies


To run this project, you need the following dependencies:

Python 3.x
pandas
scikit-learn
imbalanced-learn (for SMOTE)
Jupyter Notebook (optional, for running the notebook)


## Usage


Load and explore the dataset.


Preprocess the data (handling duplicates, dropping irrelevant features, etc.).

Address class imbalance using undersampling and SMOTE.

Train and evaluate a logistic regression model.


## Model Description


The project uses Logistic Regression, a simple yet effective algorithm for binary classification, to detect fraudulent transactions. The model's performance is evaluated using accuracy, and additional metrics like precision, recall, and F1-score can be explored further.


Key Steps:


Data Preprocessing:

Removing duplicates and irrelevant columns.

Handling class imbalance with undersampling and SMOTE.

Scaling features using StandardScaler.


Model Training:

Using Logistic Regression to fit the training data.

Training on both under-sampled and SMOTE-over-sampled datasets.

Model Evaluation:


Predicting on the test set and evaluating accuracy.

Potential extension to include precision, recall, and F1-score.


## Results


The final model achieves an accuracy of 0.94% on the test data after addressing the class imbalance. The use of SMOTE significantly improves the model's ability to correctly classify fraudulent transactions.


## Acknowledgements:

The dataset used in this project is provided by [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data).

Special thanks to the contributors of scikit-learn and imbalanced-learn libraries for their excellent tools.
