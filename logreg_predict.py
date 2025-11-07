import pandas as pd
import numpy as np
import sys
import json

from logreg_train import OneVsAllClassifier


class LogisticRegression:
    def __init__(self, weights=None, bias=None):
        self.weights = np.array(weights) if weights is not None else None
        self.bias = bias
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


class OneVsAllClassifier:
    def __init__(self):
        self.classifiers = {}
        self.classes = None
        self.feature_names = None
        
    def load_weights(self, filename):
        """Load classifier weights from file"""
        with open(filename, 'r') as f:
            weights_data = json.load(f)
        
        self.classes = np.array(weights_data['classes'])
        self.feature_names = weights_data['feature_names']
        
        # Load each classifier
        for class_label, classifier_data in weights_data['classifiers'].items():
            classifier = LogisticRegression(
                weights=classifier_data['weights'],
                bias=classifier_data['bias']
            )
            classifier.feature_names = classifier_data['feature_names']
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


def preprocess_data(df, feature_names):
    from sklearn.preprocessing import StandardScaler
    """Preprocess the dataset for prediction"""
    # Select only the features that were used in training
    available_features = [col for col in feature_names if col in df.columns]
    
    if len(available_features) != len(feature_names):
        missing_features = set(feature_names) - set(available_features)
        print(f"Warning: Missing features: {missing_features}")
    
    X = df[available_features].copy()
    
    # Handle missing values by filling with mean
    for column in X.columns:
        if X[column].isna().any():
            mean_value = X[column].mean()
            X.loc[:, column] = X[column].fillna(mean_value)
    scaler_custom = StandardScaler()
    X[X.columns] = scaler_custom.fit_transform(X[X.columns])
    return X


def predict_houses(df, weights_file):
    """Main prediction function"""
    
    # Load the trained classifier
    classifier = OneVsAllClassifier()
    classifier.load_weights(weights_file)
    
    
    
    # Preprocess features
    X = preprocess_data(df, classifier.feature_names)
    
    
    
    
    # Make predictions
    predictions = classifier.predict(X)
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'Index': range(len(predictions)),
        'Hogwarts House': predictions
    })
    
    # Save to houses.csv
    results_df.to_csv('houses.csv', index=False)
    
    return results_df


def main():
    try:
        if len(sys.argv) != 3:
            raise Exception("Number of arguments is incorrect. Usage: python logreg_predict.py dataset_test.csv weights.json")
        
        dataset_file = sys.argv[1]
        weights_file = sys.argv[2]
        
        # Check if files exist
        import os
        if not os.path.exists(dataset_file):
            raise Exception(f"Dataset file '{dataset_file}' not found")
        
        if not os.path.exists(weights_file):
            raise Exception(f"Weights file '{weights_file}' not found")
        
        # Load test dataset
        try:
            df = pd.read_csv(dataset_file)
        except Exception as e:
            raise Exception(f"Error reading CSV file: {e}")
        
        # Check if dataset is empty
        if df.empty:
            raise Exception("Test dataset is empty")
        
        
        # Make predictions
        results = predict_houses(df, weights_file)
        print("Predictions saved to 'houses.csv'")

        
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
