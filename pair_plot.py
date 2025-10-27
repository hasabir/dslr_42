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


def create_pair_plot(df):
    """Create a pair plot (scatter plot matrix) for all numerical features"""
    numerical_df = df.select_dtypes(include=['float64', 'int64'])
    
    if numerical_df.empty:
        print("No numerical columns found")
        return
    
    features = numerical_df.columns.tolist()
    n_features = len(features)
    
    if n_features == 0:
        print("No features to plot")
        return
    
    # Create subplots
    fig, axes = plt.subplots(n_features, n_features, figsize=(15, 15))
    
    # If only one feature, make it a 1D array
    if n_features == 1:
        axes = np.array([[axes]])
    
    # Plot each pair of features
    for i in range(n_features):
        for j in range(n_features):
            ax = axes[i, j] if n_features > 1 else axes[0, 0]
            
            if i == j:
                # Diagonal: histogram
                feature_data = numerical_df[features[i]].dropna()
                ax.hist(feature_data, bins=20, alpha=0.7, color='skyblue')
                ax.set_title(f'{features[i]}', fontsize=10)
            else:
                # Off-diagonal: scatter plot
                feature1 = features[i]
                feature2 = features[j]
                
                # Get common data points
                common_mask = ~(pd.isna(numerical_df[feature1]) | pd.isna(numerical_df[feature2]))
                data1 = numerical_df[feature1][common_mask]
                data2 = numerical_df[feature2][common_mask]
                
                if len(data1) > 0:
                    ax.scatter(data1, data2, alpha=0.6, s=20)
                    
                    # Calculate and display correlation
                    correlation = calculate_correlation(data1.values, data2.values)
                    ax.text(0.05, 0.95, f'r={correlation:.3f}', 
                           transform=ax.transAxes, fontsize=8,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                ax.set_xlabel(feature1, fontsize=8)
                ax.set_ylabel(feature2, fontsize=8)
            
            # Format axes
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle('Pair Plot - Feature Relationships', fontsize=16, y=0.98)
    
    # Save plot to file instead of showing
    filename = 'pair_plot_matrix.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Pair plot saved as {filename}")
    plt.close()  # Close the figure to free memory


def analyze_features_for_logistic_regression(df):
    """Analyze features to recommend which ones to use for logistic regression"""
    numerical_df = df.select_dtypes(include=['float64', 'int64'])
    
    if numerical_df.empty:
        print("No numerical columns found")
        return
    
    features = numerical_df.columns.tolist()
    correlations = {}
    
    print("\n=== Feature Analysis for Logistic Regression ===")
    print("Analyzing correlations between features...")
    
    # Calculate correlations between all pairs
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            feature1 = features[i]
            feature2 = features[j]
            
            # Get common data points
            common_mask = ~(pd.isna(numerical_df[feature1]) | pd.isna(numerical_df[feature2]))
            data1 = numerical_df[feature1][common_mask]
            data2 = numerical_df[feature2][common_mask]
            
            if len(data1) > 1:
                correlation = abs(calculate_correlation(data1.values, data2.values))
                correlations[(feature1, feature2)] = correlation
    
    # Sort correlations by strength
    sorted_correlations = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    
    print("\nTop correlations:")
    for (feat1, feat2), corr in sorted_correlations[:5]:
        print(f"{feat1} - {feat2}: {corr:.4f}")
    
    # Recommend features based on correlation analysis
    print("\n=== Recommendations for Logistic Regression ===")
    print("Based on correlation analysis:")
    
    # Find features with low correlation to each other (good for logistic regression)
    low_corr_features = []
    for feature in features:
        max_corr = 0
        for other_feature in features:
            if feature != other_feature:
                pair = tuple(sorted([feature, other_feature]))
                if pair in correlations:
                    max_corr = max(max_corr, correlations[pair])
        
        if max_corr < 0.7:  # Threshold for low correlation
            low_corr_features.append((feature, max_corr))
    
    # Sort by maximum correlation (ascending)
    low_corr_features.sort(key=lambda x: x[1])
    
    print("Recommended features (low correlation with others):")
    for feature, max_corr in low_corr_features[:6]:  # Top 6 features
        print(f"- {feature} (max correlation: {max_corr:.4f})")
    
    # Also check variance of each feature
    print("\nFeature variance analysis:")
    variances = {}
    for feature in features:
        data = numerical_df[feature].dropna()
        if len(data) > 1:
            variance = Statistic.var(data.values)
            variances[feature] = variance
    
    sorted_variances = sorted(variances.items(), key=lambda x: x[1], reverse=True)
    print("Features with highest variance (more discriminative):")
    for feature, variance in sorted_variances[:5]:
        print(f"- {feature}: {variance:.4f}")


def pair_plot(df):
    """Main function to create pair plot and analyze features"""
    print("Creating pair plot...")
    create_pair_plot(df)
    
    print("\nAnalyzing features for logistic regression...")
    analyze_features_for_logistic_regression(df)


def main():
    try:
        if len(sys.argv) != 2:
            raise Exception("Number of arguments is incorrect. Usage: python pair_plot.py dataset.csv")
        
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
            raise Exception("Need at least 2 numerical features for pair plot")
        
        print(f"Found {len(numerical_df.columns)} numerical features")
        
        pair_plot(df)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
