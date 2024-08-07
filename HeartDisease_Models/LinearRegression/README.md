# Heart Disease Prediction

This repository contains a Jupyter Notebook that demonstrates a step-by-step approach to building a logistic regression model to predict heart disease using the `heart.csv` dataset.

## Table of Contents

- [Introduction](#introduction)
- [Data Overview](#data-overview)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Data Preprocessing](#data-preprocessing)
- [Training the Model](#training-the-model)
- [Evaluating the Model](#evaluating-the-model)
- [Confusion Matrix](#confusion-matrix)
- [ROC Curve](#roc-curve)
- [Conclusion](#conclusion)
- [Installation and Usage](#installation-and-usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project aims to predict the presence of heart disease in patients using logistic regression. The dataset `heart.csv` contains 13 features and a target variable that indicates the presence or absence of heart disease.

## Data Overview

The dataset used in this project contains the following features:
- Age
- Sex
- Chest Pain Type (4 values)
- Resting Blood Pressure
- Serum Cholesterol in mg/dl
- Fasting Blood Sugar > 120 mg/dl
- Resting Electrocardiographic Results (values 0, 1, 2)
- Maximum Heart Rate Achieved
- Exercise Induced Angina
- ST Depression Induced by Exercise Relative to Rest
- Slope of the Peak Exercise ST Segment
- Number of Major Vessels (0-3) Colored by Flourosopy
- Thal (3 = normal; 6 = fixed defect; 7 = reversible defect)

The target variable is `target`, where 1 indicates the presence of heart disease and 0 indicates its absence.

## Exploratory Data Analysis (EDA)

We perform exploratory data analysis to visualize the distribution of features and their relationships with the target variable.

## Data Preprocessing

The data is split into training and testing sets using an 80/20 split. The features and target variable are separated, and logistic regression is used as the model.

## Training the Model

A logistic regression model is trained using the training data. The model is then used to predict the test data.

## Evaluating the Model

The model's performance is evaluated using accuracy, precision, recall, and F1 score metrics.

## Confusion Matrix

A confusion matrix is plotted to provide a detailed view of the model's performance in terms of true positives, true negatives, false positives, and false negatives.

## ROC Curve

The ROC curve is plotted to visualize the trade-off between the true positive rate and the false positive rate. The area under the ROC curve (AUC) is also computed.

## Conclusion

The logistic regression model shows good performance in predicting heart disease. Further tuning and testing can improve its accuracy.

## Dependencies

The following packages and their versions are required to run the notebook:

- pandas==1.5.0
- scikit-learn==1.2.0
- matplotlib==3.6.0
- seaborn==0.12.0

