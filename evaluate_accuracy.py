
import pandas as pd
import sys
from sklearn.metrics import accuracy_score


def compare_models(dataset_path, predictions_file, test_size=0.2, random_state=42):
    try:
        true_df = pd.read_csv(dataset_path)
        y_true = true_df['Hogwarts House'].values
        predectd_df = pd.read_csv(predictions_file)
        y_pred = predectd_df['Hogwarts House'].values
        if len(y_true) != len(y_pred):
            print("Error: The number of predictions does not match the number of true labels.")
            return
        accuracy = accuracy_score(y_true, y_pred)
        print("="*80)
        print("EVALUATION OF PREDICTIONS")
        print("="*80)
        print(f"Accuracy: {accuracy:.4f}\n")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python evaluate_accuracy.py <dataset_path> <predictions_file>")
        sys.exit(1)

    dataset_path = sys.argv[1]
    predictions_file = sys.argv[2]

    compare_models(dataset_path, predictions_file)