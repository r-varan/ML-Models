import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_data(filepath):
    """
    Load dataset from a CSV file.
    
    Args:
    filepath (str): Path to the CSV file.
    
    Returns:
    pd.DataFrame: Loaded dataset.
    """
    return pd.read_csv(filepath)

def preprocess_data(df):
    """
    Preprocess the dataset by separating features and target variable.
    
    Args:
    df (pd.DataFrame): The dataframe containing the dataset.
    
    Returns:
    X (pd.DataFrame): Features.
    y (pd.Series): Target variable.
    """
    X = df.iloc[:, 0:13]
    y = df["target"]
    return X, y

def split_data(X, y, test_size=0.2, random_state=23):
    """
    Split the data into training and testing sets.
    
    Args:
    X (pd.DataFrame): Features.
    y (pd.Series): Target variable.
    test_size (float): Proportion of the dataset to include in the test split.
    random_state (int): Random seed.
    
    Returns:
    X_train, X_test, y_train, y_test: Split data.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_model(X_train, y_train):
    """
    Train a logistic regression model.
    
    Args:
    X_train (pd.DataFrame): Training features.
    y_train (pd.Series): Training target variable.
    
    Returns:
    model: Trained logistic regression model.
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on the test data.
    
    Args:
    model: Trained logistic regression model.
    X_test (pd.DataFrame): Test features.
    y_test (pd.Series): Test target variable.
    
    Returns:
    dict: A dictionary with accuracy, precision, recall, and f1 score.
    """
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred)
    }
    return metrics

def main():
    # Load data
    filepath = "heart.csv"
    df = load_data(filepath)
    
    # Preprocess data
    X, y = preprocess_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    
    # Print metrics
    print(f"Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"Precision: {metrics['precision']*100:.2f}%")
    print(f"Recall: {metrics['recall']*100:.2f}%")
    print(f"F1 Score: {metrics['f1_score']*100:.2f}%")

if __name__ == "__main__":
    main()
