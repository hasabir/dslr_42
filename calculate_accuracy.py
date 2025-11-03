
import pandas as pd
import numpy as np
import sys
import time
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json

from logreg_train import OneVsAllClassifier, preprocess_data


def compare_models(dataset_path, weights_file, test_size=0.2, random_state=42):
    """Compare custom implementation with sklearn"""
    
    print("="*80)
    print("LOGISTIC REGRESSION COMPARISON: Custom vs Sklearn")
    print("="*80)

    df = pd.read_csv(dataset_path)

    X = preprocess_data(df)
    y = df['Hogwarts House'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    
    # Scale features
    scaler_custom = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler_custom.fit_transform(X_train),
        columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler_custom.transform(X_test),
        columns=X_test.columns
    )
    
    start_time = time.time()
    custom_model = OneVsAllClassifier(learning_rate=0.01, max_iterations=1000)
    custom_model.fit(X_train_scaled, y_train)
    custom_train_time = time.time() - start_time
    
    y_pred_custom = custom_model.predict(X_test_scaled)
    custom_accuracy = accuracy_score(y_test, y_pred_custom)
    
    print(f"\nTraining time: {custom_train_time:.4f} seconds")
    print(f"Test Accuracy: {custom_accuracy:.4f}")
    
    # ========== SKLEARN IMPLEMENTATION ==========
    print("\n" + "="*80)
    print("SKLEARN IMPLEMENTATION")
    print("="*80)
    
    # Sklearn automatically scales internally in some solvers, but we'll scale for fair comparison
    scaler_sklearn = StandardScaler()
    X_train_scaled_sk = scaler_sklearn.fit_transform(X_train)
    X_test_scaled_sk = scaler_sklearn.transform(X_test)
    
    start_time = time.time()
    sklearn_model = SklearnLogisticRegression(
        max_iter=1000,
        multi_class='ovr',  # One-vs-Rest (same as your implementation)
        solver='lbfgs',
        random_state=random_state
    )
    sklearn_model.fit(X_train_scaled_sk, y_train)
    sklearn_train_time = time.time() - start_time
    
    y_pred_sklearn = sklearn_model.predict(X_test_scaled_sk)
    sklearn_accuracy = accuracy_score(y_test, y_pred_sklearn)
    
    print(f"\nTraining time: {sklearn_train_time:.4f} seconds")
    print(f"Test Accuracy: {sklearn_accuracy:.4f}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python calculate_accuracy.py dataset_train.csv weights.json")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    weights_file = sys.argv[2]
    try:
        results = compare_models(dataset_path, weights_file)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()