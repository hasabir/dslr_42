import pandas as pd
import sys
import numpy as np

from typing import Any, Callable, Dict, List
from statistic import Statistic


def get_statistics(key, args):
    stats = {
        'Count': len(args),
        'Mean': Statistic.mean(args),
        'Std': Statistic.std(args),
        'Min': Statistic.min(args),
        '25%': Statistic.quantile(args, 0.25),
        '50%': Statistic.quantile(args, 0.5),
        '75%': Statistic.quantile(args, 0.75),
        'Max': Statistic.max(args),
        'variance': Statistic.variance(args),
        'median': Statistic.median(args),
    }
    return f"{stats[key]:.6f}"

def describe(df):
    # Select only numerical columns
    numerical_df = df.select_dtypes(include=['float64', 'int64'])
    
    if numerical_df.empty:
        print("No numerical columns found in the dataset")
        return
    
    # Initialize the result DataFrame
    indexes = ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max", "variance", "median"]
    result_df = pd.DataFrame(index=indexes, columns=numerical_df.columns)
    
    # Calculate statistics for each numerical column
    for column_name in numerical_df.columns:
        column_data = numerical_df[column_name].dropna().values.astype(float)
        
        if len(column_data) == 0:
            continue
            
        for stat_name in indexes:
            result_df.loc[stat_name, column_name] = get_statistics(stat_name, column_data)
    
    # Format the output to match the required format
    print(result_df.to_string(float_format='%.6f'))



def main():
    try:
        if len(sys.argv) != 2:
            raise Exception("Number of arguments is incorrect. Usage: python describe.py dataset.csv")
        
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

        # Print only the statistics table to match the required output format
        describe(df)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()