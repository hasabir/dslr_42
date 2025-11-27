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
    
    if n_features < 2:
        print("Need at least 2 features for pair plot")
        return
    
    # House colors
    house_colors = {
        'Gryffindor': '#AE0001',      # Dark red
        'Hufflepuff': '#F0C75E',      # Yellow/gold
        'Ravenclaw': '#222F5B',       # Dark blue
        'Slytherin': '#2A623D'        # Dark green
    }
    
    # Check if we have house information
    has_houses = 'Hogwarts House' in df.columns
    if has_houses:
        houses = df['Hogwarts House']
        unique_houses = sorted(houses.unique())
    
    # Create subplots
    fig, axes = plt.subplots(n_features, n_features, figsize=(15, 15))
    
    # Plot each pair of features
    for i in range(n_features):
        for j in range(n_features):
            ax = axes[i, j]
            
            if i == j:
                # Diagonal: histogram colored by house
                if has_houses:
                    house_data_list = []
                    colors_list = []
                    for house in unique_houses:
                        house_mask = houses == house
                        house_data = numerical_df[features[i]][house_mask].dropna()
                        if len(house_data) > 0:
                            house_data_list.append(house_data)
                            colors_list.append(house_colors.get(house, 'gray'))
                    
                    if house_data_list:
                        ax.hist(house_data_list, bins=15, alpha=0.7, 
                               color=colors_list, stacked=False)
                else:
                    feature_data = numerical_df[features[i]].dropna()
                    ax.hist(feature_data, bins=20, alpha=0.7, color='skyblue')
                
                if j == 0:
                    ax.set_ylabel(features[i], fontsize=8)
                if i == n_features - 1:
                    ax.set_xlabel(features[i], fontsize=8)
            else:
                # Off-diagonal: scatter plot colored by house
                feature1 = features[j]  # x-axis
                feature2 = features[i]  # y-axis
                
                # Get common data points
                common_mask = ~(pd.isna(numerical_df[feature1]) | pd.isna(numerical_df[feature2]))
                
                if has_houses:
                    common_mask = common_mask & ~pd.isna(houses)
                
                if common_mask.sum() > 0:
                    if has_houses:
                        # Plot each house with its color
                        for house in unique_houses:
                            house_mask = (houses == house) & common_mask
                            if house_mask.sum() > 0:
                                ax.scatter(numerical_df[feature1][house_mask], 
                                          numerical_df[feature2][house_mask],
                                          alpha=0.6, s=5,
                                          color=house_colors.get(house, 'gray'),
                                          edgecolors='none')
                    else:
                        data1 = numerical_df[feature1][common_mask]
                        data2 = numerical_df[feature2][common_mask]
                        ax.scatter(data1, data2, alpha=0.6, s=5)
                
                # Remove labels except on edges
                if j == 0:
                    ax.set_ylabel(features[i], fontsize=8)
                else:
                    ax.set_ylabel('')
                    
                if i == n_features - 1:
                    ax.set_xlabel(features[j], fontsize=8)
                else:
                    ax.set_xlabel('')
            
            # Format axes
            ax.tick_params(labelsize=6)
            if i != j:
                ax.grid(True, alpha=0.2)
    
    # Add legend
    if has_houses:
        handles = [plt.Line2D([0], [0], marker='o', color='w', 
                             markerfacecolor=house_colors.get(house, 'gray'), 
                             markersize=8, label=house)
                  for house in unique_houses]
        fig.legend(handles=handles, loc='upper right', fontsize=10, 
                  bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    plt.suptitle('Pair Plot - Feature Relationships', fontsize=16, y=0.995)
    
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
        return None
    
    features = numerical_df.columns.tolist()
    correlations = {}
    
    print("\n" + "="*60)
    print("FEATURE ANALYSIS FOR LOGISTIC REGRESSION")
    print("="*60)
    print("\nAnalyzing correlations between features...")
    
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
    
    print("\nTop correlations (high correlation = redundant features):")
    print("-" * 60)
    for (feat1, feat2), corr in sorted_correlations[:5]:
        print(f"{feat1:30s} - {feat2:30s}: {corr:.4f}")
    
    # Score each feature based on correlation and variance
    feature_scores = []
    for feature in features:
        # Find max correlation with any other feature
        max_corr = 0
        for other_feature in features:
            if feature != other_feature:
                pair = tuple(sorted([feature, other_feature]))
                max_corr = max(max_corr, correlations.get(pair, 0))
        
        # Calculate variance
        data = numerical_df[feature].dropna()
        variance = Statistic.variance(data.values) if len(data) > 1 else 0
        
        # Score: higher variance and lower correlation = better
        score = variance / (1 + max_corr)
        feature_scores.append((feature, max_corr, variance, score))
    
    # Sort by score (descending) - best features first
    feature_scores.sort(key=lambda x: x[3], reverse=True)
    
    # Select top features (typically 6-8 features work well for logistic regression)
    selected_features = [feat[0] for feat in feature_scores[:8]]
    
    print("\n" + "="*60)
    print("RECOMMENDED FEATURES FOR LOGISTIC REGRESSION")
    print("="*60)
    print(f"\n{'Feature':30s} | {'Max Corr':>10s} | {'Variance':>12s} | {'Score':>10s}")
    print("-" * 60)
    for i, (feature, max_corr, variance, score) in enumerate(feature_scores[:8], 1):
        print(f"{feature:30s} | {max_corr:10.4f} | {variance:12.4f} | {score:10.4f}")
    
    print(f"\nTotal: {len(selected_features)} features selected")
    print("\nRationale:")
    print("- Low correlation with other features (avoid multicollinearity)")
    print("- High variance (discriminative power for classification)")
    
    return selected_features


def pair_plot(df):
    """Main function to create pair plot and analyze features"""
    print("\n" + "="*60)
    print("CREATING PAIR PLOT")
    print("="*60)
    print("Creating scatter plot matrix (pair plot)...")
    create_pair_plot(df)
    
    print("\n" + "="*60)
    print("ANALYZING FEATURES FOR LOGISTIC REGRESSION")
    print("="*60)
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
