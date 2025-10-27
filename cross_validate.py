#!/usr/bin/env python3
"""
Cross-validation to evaluate the logistic regression model
Uses k-fold cross-validation on the training set.
"""
import pandas as pd
import numpy as np
import sys
from logreg_train import OneVsAllClassifier, preprocess_data


def cross_validate(df, k=5):
    """Perform k-fold cross-validation"""
    
    # Preprocess data
    X = preprocess_data(df)
    y = df['Hogwarts House'].values
    
    # Get indices
    indices = np.arange(len(y))
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(indices)
    
    # Split into k folds
    fold_size = len(indices) // k
    accuracies = []
    
    print(f"Performing {k}-fold cross-validation...")
    print("=" * 60)
    
    for i in range(k):
        print(f"\nFold {i + 1}/{k}")
        print("-" * 60)
        
        # Create train and validation split
        val_start = i * fold_size
        val_end = (i + 1) * fold_size if i < k - 1 else len(indices)
        
        val_indices = indices[val_start:val_end]
        train_indices = np.concatenate([indices[:val_start], indices[val_end:]])
        
        # Split data
        X_train = X.iloc[train_indices]
        y_train = y[train_indices]
        X_val = X.iloc[val_indices]
        y_val = y[val_indices]
        
        print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
        
        # Train model
        classifier = OneVsAllClassifier(learning_rate=0.01, max_iterations=500)
        classifier.fit(X_train, y_train)
        
        # Predict on validation set
        predictions = classifier.predict(X_val)
        
        # Calculate accuracy
        correct = np.sum(predictions == y_val)
        accuracy = (correct / len(y_val)) * 100
        accuracies.append(accuracy)
        
        print(f"Validation Accuracy: {accuracy:.2f}% ({correct}/{len(y_val)})")
        
        # Per-class accuracy
        classes = np.unique(y_val)
        for house in classes:
            house_mask = y_val == house
            if np.sum(house_mask) > 0:
                house_acc = np.sum(predictions[house_mask] == y_val[house_mask]) / np.sum(house_mask) * 100
                print(f"  {house}: {house_acc:.1f}%")
    
    # Summary
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Mean Accuracy: {np.mean(accuracies):.2f}% ± {np.std(accuracies):.2f}%")
    print(f"Min Accuracy:  {np.min(accuracies):.2f}%")
    print(f"Max Accuracy:  {np.max(accuracies):.2f}%")
    print()
    
    for i, acc in enumerate(accuracies, 1):
        print(f"Fold {i}: {acc:.2f}%")


def main():
    try:
        dataset_path = 'dataset_train.csv'
        
        import os
        if not os.path.exists(dataset_path):
            raise Exception(f"File '{dataset_path}' not found")
        
        # Load dataset
        df = pd.read_csv(dataset_path)
        
        if df.empty:
            raise Exception("Dataset is empty")
        
        if 'Hogwarts House' not in df.columns:
            raise Exception("'Hogwarts House' column not found in dataset")
        
        print(f"Loaded dataset with {len(df)} samples")
        print()
        
        # Perform cross-validation
        cross_validate(df, k=5)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
