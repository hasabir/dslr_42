import pandas as pd
import numpy as np
import time
import sys
import json
from statistic import Statistic
from logreg_train import LogisticRegression
from logreg_train import preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from logreg_train import OneVsAllClassifier

class LogisticRegressionBonus(LogisticRegression):
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6, weights=None, bias=0):
        super().__init__(learning_rate, max_iterations, tolerance)
        self.weights = weights
        self.bias = bias

    def fit(self, X, y):
        m, n = X.shape

        if self.weights is None:
            self.weights = np.zeros(n)
        if self.bias is None:
            self.bias = 0

        previous_cost = float('inf')

        for iteration in range(self.max_iterations):
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(m):
                xi = X_shuffled[i].reshape(1, -1)
                yi = y_shuffled[i]

                prediction = self.predict_probability(xi)
                dw = np.dot(xi.T, (prediction - yi))
                db = prediction - yi

                self.weights -= self.learning_rate * dw.flatten()
                self.bias -= self.learning_rate * db

            cost = self.compute_cost(X, y)

            if abs(previous_cost - cost) < self.tolerance:
                print(f"Converged after {iteration + 1} epochs")
                break

            previous_cost = cost

            if (iteration + 1) % 100 == 0:
                print(f"Epoch {iteration + 1}, Cost: {cost:.6f}")

        print(f"Final Cost: {cost:.6f}")


class OneVsAllClassifierBonus(OneVsAllClassifier):
    def __init__(self, learning_rate=0.01, max_iterations=1000):
        super().__init__(learning_rate, max_iterations)

    def load_weights(self, filename):
        with open(filename, 'r') as f:
            weights_data = json.load(f)

        self.classes = np.array(weights_data['classes'])
        self.feature_names = weights_data['feature_names']

        for class_label, classifier_data in weights_data['classifiers'].items():
            classifier = LogisticRegressionBonus(
                weights=np.array(classifier_data['weights']),
                bias=classifier_data['bias'],
                learning_rate=self.learning_rate,
                max_iterations=self.max_iterations
            )
            classifier.feature_names = classifier_data['feature_names']
            self.classifiers[class_label] = classifier

        print(f"Loaded weights for {len(self.classes)} classes: {self.classes}")


def train_logistic_regression(df):
    from sklearn.preprocessing import StandardScaler

    """Main training function"""
    print("Preprocessing data...")
    
    # Preprocess features
    X = preprocess_data(df)

    
    if 'Hogwarts House' not in df.columns:
        raise ValueError("'Hogwarts House' column not found in dataset")
    print('*' * 50)
    
    y = df['Hogwarts House'].values
    
    # Create and train one-vs-all classifier
    classifier = OneVsAllClassifierBonus(learning_rate=0.01, max_iterations=1000)
    classifier.fit(X, y)
    
    # Save weights
    classifier.save_weights('weights.json')
    
    print("\nTraining completed successfully!")
    return classifier



def calculate_accuracy(dataset_path, weights_file, test_size=0.2, random_state=42):
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
    custom_model = OneVsAllClassifierBonus(learning_rate=0.0001, max_iterations=1000)
    custom_model.fit(X_train_scaled, y_train)
    custom_train_time = time.time() - start_time
    
    y_pred_custom = custom_model.predict(X_test_scaled)
    custom_accuracy = accuracy_score(y_test, y_pred_custom)
    
    print(f"\nTraining time: {custom_train_time:.4f} seconds")
    print(f"Test Accuracy: {custom_accuracy:.4f}")



def main():
    try:
        if len(sys.argv) != 2:
            raise Exception("Number of arguments is incorrect. Usage: python logreg_train.py dataset_train.csv")
        
        dataset_path = sys.argv[1]
        
        # Check if file exists
        import os
        if not os.path.exists(dataset_path):
            raise Exception(f"File '{dataset_path}' not found")
        
        # Load dataset
        try:
            df = pd.read_csv(dataset_path)
        except Exception as e:
            raise Exception(f"Error reading CSV file: {e}")
        
        # Check if dataset is empty
        if df.empty:
            raise Exception("Dataset is empty")
        
        # Check if Hogwarts House column exists
        if 'Hogwarts House' not in df.columns:
            raise Exception("'Hogwarts House' column not found in dataset")
        
        # Check for numerical columns
        numerical_df = df.select_dtypes(include=['float64', 'int64'])
        if numerical_df.empty:
            raise Exception("No numerical columns found in dataset")
        
        print(f"Dataset loaded successfully. Shape: {df.shape}")
        
        # Train the model
        classifier = train_logistic_regression(df)
        calculate_accuracy(dataset_path, 'weights.json')
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
