import pandas as pd
import numpy as np
import sys
import json
from statistic import Statistic


class LogisticRegression:
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.weights = None
        self.bias = None
        self.feature_names = None
        
    def sigmoid(self, z):
        """Sigmoid activation function"""
        # Clip z to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def predict_probability(self, X):
        """Predict probability using sigmoid function"""
        if self.weights is None or self.bias is None:
            raise ValueError("Model not trained yet")
        
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)
    
    def predict(self, X):
        """Predict binary class (0 or 1)"""
        probabilities = self.predict_probability(X)
        return (probabilities >= 0.5).astype(int)
    
    def compute_cost(self, X, y):
        """Compute logistic regression cost function"""
        m = len(y)
        predictions = self.predict_probability(X)
        
        # Avoid log(0) by clipping predictions
        predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
        
        cost = -(1/m) * np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
        return cost
    
    def compute_gradients(self, X, y):
        """Compute gradients for logistic regression"""
        m = len(y)
        predictions = self.predict_probability(X)
        
        dw = (1/m) * np.dot(X.T, (predictions - y))
        db = (1/m) * np.sum(predictions - y)
        
        return dw, db
    
    def fit(self, X, y):
        """Train the logistic regression model using gradient descent"""
        m, n = X.shape
        
        # Initialize weights and bias
        self.weights = np.zeros(n)
        self.bias = 0
        
        previous_cost = float('inf')
        
        for iteration in range(self.max_iterations):
            # Forward propagation
            predictions = self.predict_probability(X)
            
            # Compute cost
            cost = self.compute_cost(X, y)
            
            # Compute gradients
            dw, db = self.compute_gradients(X, y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Check for convergence
            if abs(previous_cost - cost) < self.tolerance:
                print(f"Converged after {iteration + 1} iterations")
                break
            
            previous_cost = cost
            
            # Print progress every 100 iterations
            if (iteration + 1) % 100 == 0:
                print(f"Iteration {iteration + 1}, Cost: {cost:.6f}")
        
        print(f"Final cost: {cost:.6f}")
    
    def get_parameters(self):
        """Get model parameters"""
        return {
            'weights': self.weights.tolist(),
            'bias': self.bias,
            'feature_names': self.feature_names
        }


class OneVsAllClassifier:
    def __init__(self, learning_rate=0.01, max_iterations=1000):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.classifiers = {}
        self.classes = None
        self.feature_names = None
        
    def fit(self, X, y):
        """Train one-vs-all classifiers"""
        self.classes = np.unique(y)
        self.feature_names = X.columns.tolist() if hasattr(X, 'columns') else None
        
        print(f"Training {len(self.classes)} binary classifiers for classes: {self.classes}")
        
        for class_label in self.classes:
            print(f"\nTraining classifier for class: {class_label}")
            
            # Create binary labels for this class
            binary_y = (y == class_label).astype(int)
            
            # Train logistic regression classifier
            classifier = LogisticRegression(
                learning_rate=self.learning_rate,
                max_iterations=self.max_iterations
            )
            # Attach feature names so individual classifiers retain the feature ordering
            classifier.feature_names = self.feature_names
            classifier.fit(X.values, binary_y)
            
            self.classifiers[class_label] = classifier
    
    def predict_probabilities(self, X):
        """Predict probabilities for all classes"""
        probabilities = {}
        
        for class_label, classifier in self.classifiers.items():
            prob = classifier.predict_probability(X.values)
            probabilities[class_label] = prob
        
        return probabilities
    
    def predict(self, X):
        """Predict class labels"""
        probabilities = self.predict_probabilities(X)
        
        # Convert to array for easier handling
        prob_array = np.array([probabilities[cls] for cls in self.classes]).T
        
        # Get class with highest probability
        predictions = self.classes[np.argmax(prob_array, axis=1)]
        
        return predictions
    
    def save_weights(self, filename):
        """Save all classifier weights to a file"""
        weights_data = {
            'classes': self.classes.tolist(),
            'feature_names': self.feature_names,
            'classifiers': {}
        }
        
        for class_label, classifier in self.classifiers.items():
            weights_data['classifiers'][str(class_label)] = classifier.get_parameters()
        
        with open(filename, 'w') as f:
            json.dump(weights_data, f, indent=2)
        
        print(f"Weights saved to {filename}")


def preprocess_data(df):
    """Preprocess the dataset for training"""
    # Select numerical features only
    numerical_df = df.select_dtypes(include=['float64', 'int64'])
    
    # Remove features that might not be useful for classification
    # (like Index if it exists)
    if 'Index' in numerical_df.columns:
        numerical_df = numerical_df.drop('Index', axis=1)
    
    # Handle missing values by filling with mean
    for column in numerical_df.columns:
        if numerical_df[column].isna().any():
            mean_value = numerical_df[column].mean()
            print(f"************************ before numerical_df[column] = {numerical_df[column]}")
            numerical_df[column] = numerical_df[column].fillna(mean_value)
            print(f"************************ after numerical_df[column] = {numerical_df[column]}")
        
    return numerical_df


def train_logistic_regression(df):
    from sklearn.preprocessing import StandardScaler

    """Main training function"""
    print("Preprocessing data...")
    
    # Preprocess features
    X = preprocess_data(df)
    # scaler_custom = StandardScaler()
    # X[X.columns] = scaler_custom.fit_transform(X[X.columns])

    print("******************************** X after preprocess_data ************************\n", X.head())
    
    # Get target variable (Hogwarts House)
    if 'Hogwarts House' not in df.columns:
        raise ValueError("'Hogwarts House' column not found in dataset")
    print('*' * 50)
    
    y = df['Hogwarts House'].values
    print("******************************** y after extracting target ************************\n", y)
    
    print(f"Training data shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    print(f"Classes: {np.unique(y)}")
    print(f"Features: {list(X.columns)}")
    
    # Create and train one-vs-all classifier
    classifier = OneVsAllClassifier(learning_rate=0.01, max_iterations=1000)
    classifier.fit(X, y)
    
    # Save weights
    classifier.save_weights('weights.json')
    
    print("\nTraining completed successfully!")
    return classifier


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
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
