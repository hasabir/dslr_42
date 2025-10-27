#!/usr/bin/env python3
"""
Evaluate the accuracy of the logistic regression classifier
by comparing predictions against actual labels in the training set.
"""
import pandas as pd
import sys
import json
import numpy as np


def evaluate_accuracy(train_file, test_file, predictions_file):
    """Evaluate prediction accuracy"""
    
    # Load datasets
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    predictions_df = pd.read_csv(predictions_file)
    
    print("=== EVALUATION REPORT ===\n")
    
    # Check if test dataset has labels (for accuracy calculation)
    if 'Hogwarts House' in test_df.columns:
        print("✓ Test dataset contains labels - calculating accuracy\n")
        
        actual = test_df['Hogwarts House'].values
        predicted = predictions_df['Hogwarts House'].values
        
        # Overall accuracy
        correct = np.sum(actual == predicted)
        total = len(actual)
        accuracy = (correct / total) * 100
        
        print(f"Overall Accuracy: {accuracy:.2f}% ({correct}/{total})")
        print()
        
        # Per-class accuracy
        classes = np.unique(actual)
        print("Per-Class Accuracy:")
        print("-" * 50)
        
        for house in classes:
            house_mask = actual == house
            house_actual = actual[house_mask]
            house_predicted = predicted[house_mask]
            
            house_correct = np.sum(house_actual == house_predicted)
            house_total = len(house_actual)
            house_accuracy = (house_correct / house_total) * 100 if house_total > 0 else 0
            
            print(f"{house:20s}: {house_accuracy:6.2f}% ({house_correct}/{house_total})")
        
        print()
        
        # Confusion matrix
        print("Confusion Matrix:")
        print("-" * 80)
        print(f"{'Actual \\ Predicted':<20s}", end='')
        for house in classes:
            print(f"{house:>15s}", end='')
        print()
        print("-" * 80)
        
        for actual_house in classes:
            print(f"{actual_house:<20s}", end='')
            for pred_house in classes:
                mask = (actual == actual_house) & (predicted == pred_house)
                count = np.sum(mask)
                print(f"{count:>15d}", end='')
            print()
        
        print()
        
    else:
        print("⚠ Test dataset does not contain labels")
        print("Cannot calculate accuracy without ground truth labels\n")
    
    # Prediction distribution
    print("Prediction Distribution:")
    print("-" * 50)
    pred_counts = predictions_df['Hogwarts House'].value_counts()
    for house, count in pred_counts.items():
        percentage = (count / len(predictions_df)) * 100
        print(f"{house:20s}: {count:4d} ({percentage:5.1f}%)")
    
    print()
    
    # Training data distribution for comparison
    print("Training Data Distribution:")
    print("-" * 50)
    train_counts = train_df['Hogwarts House'].value_counts()
    for house, count in train_counts.items():
        percentage = (count / len(train_df)) * 100
        print(f"{house:20s}: {count:4d} ({percentage:5.1f}%)")


def main():
    try:
        train_file = 'dataset_train.csv'
        test_file = 'dataset_test.csv'
        predictions_file = 'houses.csv'
        
        import os
        for file in [train_file, test_file, predictions_file]:
            if not os.path.exists(file):
                raise Exception(f"File '{file}' not found")
        
        evaluate_accuracy(train_file, test_file, predictions_file)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
