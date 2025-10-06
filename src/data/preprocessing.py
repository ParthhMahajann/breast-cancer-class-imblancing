
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

def stratified_split(X, y, test_size=0.2, random_state=42):
    """
    Perform stratified train-test split.
    
    Args:
        X: feature matrix
        y: target vector
        test_size: fraction for test set
        random_state: random seed
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

def scale_features(X_train, X_test):
    """
    Scale features using StandardScaler fitted on training data.
    
    Args:
        X_train: training feature matrix
        X_test: test feature matrix
        
    Returns:
        tuple: (X_train_scaled, X_test_scaled, scaler)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler

def verify_scaling(X_scaled, feature_names=None, num_features=5):
    """
    Verify that scaling was applied correctly.
    
    Args:
        X_scaled: scaled feature matrix
        feature_names: optional feature names
        num_features: number of features to check
    """
    print("Scaling Verification:")
    for i in range(min(num_features, X_scaled.shape[1])):
        feature_name = feature_names[i] if feature_names else f"Feature {i}"
        mean = X_scaled[:, i].mean()
        std = X_scaled[:, i].std()
        print(f"{feature_name}: Mean={mean:.6f}, Std={std:.6f}")