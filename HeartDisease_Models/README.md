# Heart Disease Prediction and Clustering Analysis

This repository contains two machine learning models applied to the heart disease dataset:
1. **Logistic Regression** for predicting the likelihood of heart disease based on various features.
2. **K-Means Clustering** for analyzing the clustering behavior of features related to heart disease.

## Table of Contents
1. [Overview](#overview)
2. [Logistic Regression](#logistic-regression)
3. [K-Means Clustering](#k-means-clustering)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Dependencies](#dependencies)

## Overview

This project involves the application of two different models to the heart disease dataset:
- **Logistic Regression**: Used to predict the presence of heart disease based on various health metrics.
- **K-Means Clustering**: Used to explore the clustering patterns of continuous features in the dataset.

## Logistic Regression

The logistic regression model aims to predict the likelihood of heart disease using several health-related features. The model is trained on a subset of the data and evaluated using metrics like accuracy, precision, recall, and F1 score.

### Key Steps
1. **Data Loading**: Load the heart disease dataset.
2. **Data Preparation**: Split data into training and testing sets.
3. **Model Training**: Fit a logistic regression model to the training data.
4. **Prediction**: Predict heart disease presence on the test data.
5. **Evaluation**: Assess model performance using accuracy, precision, recall, and F1 score.

## K-Means Clustering

The K-Means clustering model is used to identify clusters within the data based on continuous features. This analysis helps understand which variables significantly influence clustering.

### Key Steps
Data Preparation: Select and scale continuous features.
Clustering: Apply K-Means clustering to the scaled data.
Visualization: Plot clusters and cluster centers to understand feature relationships.

## Dependencies

Ensure you have the following Python packages installed:

pandas==1.5.0
scikit-learn==1.2.0
matplotlib==3.6.0


