import pandas as pd
from scipy.stats import f_oneway
import sys
import numpy as np
from describe import describe
from statistic import Statistic
import matplotlib.pyplot as plt



def count_runs(signs):
    runs = 1
    for i in range(1, len(signs)):
        if signs[i] != signs[i-1]:
            runs += 1
    return runs


def plot_histogram_for_course(course_name, data):
    plt.figure(figsize=(10, 6))
    houses = data['Hogwarts House'].unique()
    
    for house in houses:
        house_data = data[data['Hogwarts House'] == house][course_name]
        plt.hist(house_data, bins=20, alpha=0.5, label=house)

    plt.title(f'Score Distribution for {course_name}')
    plt.xlabel('Scores')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Save plot to file instead of showing
    filename = f'histogram_{course_name.replace(" ", "_")}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Histogram saved as {filename}")
    plt.close()  # Close the figure to free memory
    
    

def histogram(df):
    dataset = df.select_dtypes(include=['float64'])
    for course in dataset.columns:
        plot_histogram_for_course(course, df)


def check_homogeneity(course_name, df):
    df_course = df[['Hogwarts House', course_name]].dropna()
    
    # Get groups for each house (ensure all houses exist in the data)
    houses = df['Hogwarts House'].unique()
    groups = []
    
    for house in houses:
        house_scores = df_course[df_course['Hogwarts House'] == house][course_name].dropna()
        # Only include non-empty groups
        if len(house_scores) > 0:
            groups.append(house_scores.values)

    # Need at least two groups with data to perform ANOVA
    if len(groups) < 2:
        print(f"Not enough groups to perform ANOVA for {course_name}")
        return 1.0

    f_stat, p_value = f_oneway(*groups)
    print(f"{course_name}: p-value = {p_value:.5f}")
    return p_value



def check_homogeneity_by_mean_and_median(course_name, df):
    """Check homogeneity by comparing means and medians across houses"""
    df_course = df[['Hogwarts House', course_name]].dropna()
    
    # Get groups for each house
    houses = df['Hogwarts House'].unique()
    
    print(f"\n=== Homogeneity Analysis for {course_name} ===")
    print("Mean and Median by House:")
    
    means = []
    medians = []
    
    for house in houses:
        house_scores = df_course[df_course['Hogwarts House'] == house][course_name]
        if len(house_scores) > 0:
            mean_val = house_scores.mean()
            median_val = house_scores.median()
            means.append(mean_val)
            medians.append(median_val)
            print(f"{house}: Mean={mean_val:.4f}, Median={median_val:.4f}")
    
    # Calculate variance of means and medians
    if len(means) > 1:
        mean_variance = np.var(means)
        median_variance = np.var(medians)
        
        print(f"\nVariance of means: {mean_variance:.6f}")
        print(f"Variance of medians: {median_variance:.6f}")
        
        # Lower variance suggests more homogeneity
        if mean_variance < 0.1 and median_variance < 0.1:
            print("Result: Houses appear homogeneous (low variance in means/medians)")
        else:
            print("Result: Houses appear heterogeneous (high variance in means/medians)")
    
    # Also perform ANOVA test for comparison
    groups = []
    for house in houses:
        house_scores = df_course[df_course['Hogwarts House'] == house][course_name].dropna()
        if len(house_scores) > 0:
            groups.append(house_scores.values)

    if len(groups) < 2:
        print("Not enough groups to perform ANOVA for comparison")
        return 1.0

    f_stat, p_value = f_oneway(*groups)
    print(f"ANOVA p-value: {p_value:.5f}")

    return p_value



def main():
    try:
        if len(sys.argv) != 2:
            raise Exception("Number of arguments is incorrect. Usage: python histogram.py dataset.csv")
        
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
        
        print(f"Dataset loaded successfully. Shape: {df.shape}")
        
        # Check for numerical columns
        numerical_df = df.select_dtypes(include=['float64', 'int64'])
        if numerical_df.empty:
            raise Exception("No numerical columns found in dataset")
        
        print(f"Found {len(numerical_df.columns)} numerical features")
        
        # Calculate homogeneity for all courses
        tmp = df.select_dtypes(include=['float64'])
        p_values = {}
        for course in tmp.columns:
            p_values[course] = check_homogeneity(course, df)
        
        # Display histograms
        histogram(df)
        
        # Show summary statistics
        test = df.groupby('Hogwarts House')['Arithmancy'].agg(['mean', 'median'])
        print("\nArithmancy statistics by house:")
        print(test)
        
        print("\nHomogeneity p-values:")
        for course, p_val in p_values.items():
            print(f"{course}: {p_val:.5f}")
        
        # Demonstrate the enhanced homogeneity analysis
        print("\n" + "="*60)
        print("ENHANCED HOMOGENEITY ANALYSIS")
        print("="*60)
        check_homogeneity_by_mean_and_median('Arithmancy', df)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()


