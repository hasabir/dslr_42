import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from statistic import Statistic


def calculate_correlation(x, y):
    """Calculate Pearson correlation coefficient manually"""
    n = len(x)
    if n == 0:
        return 0
    
    mean_x = Statistic.mean(x)
    mean_y = Statistic.mean(y)
    
    numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    
    sum_sq_x = sum((x[i] - mean_x) ** 2 for i in range(n))
    sum_sq_y = sum((y[i] - mean_y) ** 2 for i in range(n))
    
    denominator = (sum_sq_x * sum_sq_y) ** 0.5
    
    if denominator == 0:
        return 0
    
    return numerator / denominator


def find_most_similar_features(df):
    """Find the two features with the highest correlation"""
    numerical_df = df.select_dtypes(include=['float64', 'int64'])
    
    if numerical_df.empty:
        print("No numerical columns found")
        return None, None, 0
    
    features = numerical_df.columns.tolist()
    max_correlation = -1
    most_similar_pair = (None, None)
    actual_correlation = 0  
    
    # Calculate correlation between all pairs of features
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            feature1 = features[i]
            feature2 = features[j]
            
            # Get data for both features, removing NaN values
            data1 = numerical_df[feature1].dropna().values
            data2 = numerical_df[feature2].dropna().values
            
            # Find common indices (where both features have non-NaN values)
            common_mask = ~(pd.isna(numerical_df[feature1]) | pd.isna(numerical_df[feature2]))
            common_data1 = numerical_df[feature1][common_mask].values
            common_data2 = numerical_df[feature2][common_mask].values
            
            if len(common_data1) < 2:
                continue
                
            correlation = calculate_correlation(common_data1, common_data2)
            
            if abs(correlation) > max_correlation:
                max_correlation = abs(correlation)
                most_similar_pair = (feature1, feature2)
                actual_correlation = correlation  
    
    return most_similar_pair[0], most_similar_pair[1], actual_correlation


def create_scatter_plot(df, feature1, feature2):
    """Create a scatter plot for the two most similar features"""
    plt.figure(figsize=(10, 8))
    
    # Get data for both features, removing NaN values
    common_mask = ~(pd.isna(df[feature1]) | pd.isna(df[feature2]))
    data1 = df[feature1][common_mask]
    data2 = df[feature2][common_mask]
    
    # Color by Hogwarts House if available
    if 'Hogwarts House' in df.columns:
        houses = df['Hogwarts House'][common_mask]
        
        # Map house names to colors
        house_colors = {
            'Gryffindor': '#AE0001',      # Dark red
            'Hufflepuff': '#F0C75E',      # Yellow/gold
            'Ravenclaw': '#222F5B',       # Dark blue
            'Slytherin': '#2A623D'        # Dark green
        }
        
        # Plot each house with its color
        for house in sorted(df['Hogwarts House'].unique()):
            mask = houses == house
            if mask.sum() > 0:
                plt.scatter(data1[mask], data2[mask], 
                           alpha=0.6, s=30, 
                           color=house_colors.get(house, 'gray'),
                           label=house, 
                           edgecolors='black', 
                           linewidth=0.5)
        
        plt.legend(loc='upper right')
    else:
        plt.scatter(data1, data2, alpha=0.6, s=50)
    
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.title(f'Scatter Plot: {feature1} vs {feature2}')
    plt.grid(True, alpha=0.3)
    
    # Add correlation coefficient to the plot
    correlation = calculate_correlation(data1.values, data2.values)
    plt.text(0.05, 0.95, f'Correlation: {correlation:.4f}', 
             transform=plt.gca().transAxes, fontsize=12,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot to file instead of showing
    filename = f'scatter_plot_{feature1.replace(" ", "_")}_vs_{feature2.replace(" ", "_")}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Scatter plot saved as {filename}")
    plt.close()  # Close the figure to free memory


def scatter_plot(df):
    """Main function to find and visualize the two most similar features"""
    feature1, feature2, correlation = find_most_similar_features(df)
    
    if feature1 is None or feature2 is None:
        print("Could not find suitable features for scatter plot")
        return
    
    print("\n" + "="*60)
    print("SCATTER PLOT ANALYSIS")
    print("="*60)
    print(f"\nThe two most similar features are: {feature1} and {feature2}")
    print(f"Correlation coefficient: {correlation:.6f}")
    # print(f"Correlation strength: {abs(correlation):.6f}")
    print(f"Direction: {'Positive (same direction)' if correlation > 0 else 'Negative (inverse relationship)'}")
    print(f"\n(A correlation close to 1.0 or -1.0 indicates strong similarity)")
    print(f"\nCreating scatter plot...")
    
    create_scatter_plot(df, feature1, feature2)
    
def main():
    try:
        if len(sys.argv) != 2:
            raise Exception("Number of arguments is incorrect. Usage: python scatter_plot.py dataset.csv")
        
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
        
        print(f"Dataset loaded successfully. Shape: {df.shape}")
        
        # Check for numerical columns
        numerical_df = df.select_dtypes(include=['float64', 'int64'])
        if numerical_df.empty:
            raise Exception("No numerical columns found in dataset")
        
        if len(numerical_df.columns) < 2:
            raise Exception("Need at least 2 numerical features for scatter plot")
        
        print(f"Found {len(numerical_df.columns)} numerical features")
        
        scatter_plot(df)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
