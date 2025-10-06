from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np

def load_breast_cancer_data():
    """
    Load and return breast cancer dataset from sklearn.
    
    Returns:
        tuple: (X, y, feature_names, target_names) where
            X: feature matrix (569, 30)
            y: target vector (569,)
            feature_names: list of feature names
            target_names: list of target class names
    """
    data = load_breast_cancer()
    return data.data, data.target, data.feature_names, data.target_names

def create_dataframe(X, y, feature_names):
    """
    Create pandas DataFrame from feature matrix and target vector.
    
    Args:
        X: feature matrix
        y: target vector  
        feature_names: list of feature names
        
    Returns:
        pd.DataFrame: DataFrame with features and target
    """
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    return df

def analyze_class_distribution(y):
    """
    Analyze and print class distribution information.
    
    Args:
        y: target vector
        
    Returns:
        dict: Dictionary with class distribution statistics
    """
    unique, counts = np.unique(y, return_counts=True)
    
    stats = {
        'total_samples': len(y),
        'num_classes': len(unique),
        'class_counts': dict(zip(unique, counts)),
        'class_percentages': {k: v/len(y)*100 for k, v in zip(unique, counts)},
        'imbalance_ratio': counts[1] / counts[0] if len(counts) == 2 else None
    }
    
    print("Class Distribution Analysis:")
    print(f"Total samples: {stats['total_samples']}")
    for class_label, count in stats['class_counts'].items():
        percentage = stats['class_percentages'][class_label]
        class_name = 'Malignant' if class_label == 0 else 'Benign'
        print(f"{class_name} ({class_label}): {count} samples ({percentage:.1f}%)")
    
    if stats['imbalance_ratio']:
        print(f"Imbalance ratio (Benign:Malignant): {stats['imbalance_ratio']:.2f}:1")
    
    return stats