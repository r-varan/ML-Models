"""
K-Nearest Neighbors (KNN) Clustering Algorithm Implementation

This script implements a KNN model to classify data points into clusters for a given dataset.
The steps include:
1. Preprocessing the data to extract relevant features and convert categorical labels to numeric form.
2. Calculating the Euclidean distance between test points and all training points.
3. Finding the k-nearest neighbors based on these distances.
4. Assigning a cluster to each test point based on the majority vote of its k-nearest neighbors.

This manual implementation is designed for educational purposes to understand the KNN algorithm better.
"""

import numpy as np
import pandas as pd

# Function 1: Preprocessing train and test dataframe input
def preprocess_data(train_df, test_df):
    '''
    Subsetting for columns relevant for algorithm. Creating a new numeric/trinary column for clusters to make indexing easier down the line. Everything is then converted to numpy array. The key assumption here is that both test and training data are in the same order, and that they both have the same amount of features.
    '''
    train_data = train_df.iloc[:, 1:3].to_numpy()
    test_data = test_df.iloc[:, 1:].to_numpy()
    
    mapping = {'A': 0, 'B': 1, 'C': 2}
    train_df['cluster'] = train_df['class'].replace(mapping).to_numpy()
    
    return train_data, test_data, train_df['cluster'].to_numpy()

# Function 2: Finding Euclidean Distance between test point and all train points
def compute_distances(train, test_point):
    """Calculates the Euclidean distance between a test data point
    and all train data points."""
    return np.sqrt(np.sum((train - test_point) ** 2, axis=1))

# Function 3: Finding the indices of the k-closest datapoints
def find_k_nearest_indices(distances, k):
    """Calculating the indices of the top k closest train datapoints."""
    return np.argsort(distances)[:k]

# Function 4: Establishing a cluster ID by majority vote of k neighbours
def assign_cluster(cluster_data, nearest_indices):
    """Assigns cluster based on majority vote of k-nearest neighbours."""
    k_clusters = cluster_data[nearest_indices]
    mode = np.bincount(k_clusters).argmax()
    return ['A', 'B', 'C'][mode]

# Function 5: Combining everything together
def test_cluster_ids(train_df, test_df, k):
    train, test, cluster_data = preprocess_data(train_df, test_df)
    
    cluster_ids = []
    for test_point in test:
        distances = compute_distances(train, test_point)
        nearest_indices = find_k_nearest_indices(distances, k)
        cluster_id = assign_cluster(cluster_data, nearest_indices)
        cluster_ids.append(cluster_id)
    
    test_df['cluster'] = cluster_ids
    return test_df

# Example usage
train_data_df = pd.read_csv('knn_train_data.csv')
test_data_df = pd.read_csv('test_data.csv')

final_output_test_df = test_cluster_ids(train_data_df, test_data_df, k=10)
print(final_output_test_df)
