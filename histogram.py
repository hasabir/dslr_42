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
    """Plot histogram for a course showing distribution across all houses"""
    plt.figure(figsize=(10, 6))
    houses = sorted(data['Hogwarts House'].unique())
    
    # Map house names to colors (matching typical Hogwarts house colors)
    house_colors = {
        'Gryffindor': '#AE0001',      # Dark red
        'Hufflepuff': '#F0C75E',      # Yellow/gold
        'Ravenclaw': '#222F5B',       # Dark blue
        'Slytherin': '#2A623D'        # Dark green
    }
    
    # Collect data for each house
    house_data_list = []
    colors_list = []
    labels_list = []
    
    for house in houses:
        house_data = data[data['Hogwarts House'] == house][course_name].dropna()
        if len(house_data) > 0:
            house_data_list.append(house_data)
            colors_list.append(house_colors.get(house, None))
            labels_list.append(house)
    
    # Plot with side-by-side bars (not overlapping)
    plt.hist(house_data_list, bins=20, alpha=0.7, label=labels_list, 
             color=colors_list, edgecolor='black')

    plt.title(f'Score Distribution for {course_name}')
    plt.xlabel('Scores')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Save plot to file instead of showing
    filename = f'histogram_{course_name.replace(" ", "_")}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Histogram saved as {filename}")
    plt.close()  # Close the figure to free memory
    
    

def histogram(df):
    """Plot histograms for all courses"""
    dataset = df.select_dtypes(include=['float64'])
    print(f"\nPlotting histograms for {len(dataset.columns)} courses...")
    for course in dataset.columns:
        plot_histogram_for_course(course, df)


def calculate_variance_of_means(course_name, df):
    """Calculate variance of means across houses as a homogeneity metric"""
    df_course = df[['Hogwarts House', course_name]].dropna()
    houses = df['Hogwarts House'].unique()
    means = []
    
    for house in houses:
        house_scores = df_course[df_course['Hogwarts House'] == house][course_name].dropna()
        if len(house_scores) > 0:
            means.append(house_scores.mean())
    
    if len(means) < 2:
        return float('inf')  # Return high value if not enough data
    
    return np.var(means)


def calculate_variance_of_medians(course_name, df):
    """Calculate variance of medians across houses as a homogeneity metric"""
    df_course = df[['Hogwarts House', course_name]].dropna()
    houses = df['Hogwarts House'].unique()
    medians = []
    
    for house in houses:
        house_scores = df_course[df_course['Hogwarts House'] == house][course_name].dropna()
        if len(house_scores) > 0:
            medians.append(house_scores.median())
    
    if len(medians) < 2:
        return float('inf')
    
    return np.var(medians)


def calculate_range_of_means(course_name, df):
    """Calculate range (max - min) of means across houses"""
    df_course = df[['Hogwarts House', course_name]].dropna()
    houses = df['Hogwarts House'].unique()
    means = []
    
    for house in houses:
        house_scores = df_course[df_course['Hogwarts House'] == house][course_name].dropna()
        if len(house_scores) > 0:
            means.append(house_scores.mean())
    
    if len(means) < 2:
        return float('inf')
    
    return max(means) - min(means)


def check_homogeneity(course_name, df):
    """Returns (p_value, f_statistic) for homogeneity analysis"""
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
        return (1.0, 0.0)

    f_stat, p_value = f_oneway(*groups)
    print(f"{course_name}: p-value = {p_value:.5f}")
    return (p_value, f_stat)



# def check_homogeneity_by_mean_and_median(course_name, df):
#     """Check homogeneity by comparing means and medians across houses"""
#     df_course = df[['Hogwarts House', course_name]].dropna()
    
#     # Get groups for each house
#     houses = df['Hogwarts House'].unique()
    
#     print(f"\n=== Homogeneity Analysis for {course_name} ===")
#     print("Mean and Median by House:")
    
#     means = []
#     medians = []
    
#     for house in houses:
#         house_scores = df_course[df_course['Hogwarts House'] == house][course_name]
#         if len(house_scores) > 0:
#             mean_val = house_scores.mean()
#             median_val = house_scores.median()
#             means.append(mean_val)
#             medians.append(median_val)
#             print(f"{house}: Mean={mean_val:.4f}, Median={median_val:.4f}")
    
#     # Calculate variance of means and medians
#     if len(means) > 1:
#         mean_variance = np.var(means)
#         median_variance = np.var(medians)
        
#         print(f"\nVariance of means: {mean_variance:.6f}")
#         print(f"Variance of medians: {median_variance:.6f}")
        
#         # Lower variance suggests more homogeneity
#         if mean_variance < 0.1 and median_variance < 0.1:
#             print("Result: Houses appear homogeneous (low variance in means/medians)")
#         else:
#             print("Result: Houses appear heterogeneous (high variance in means/medians)")
    
#     # Also perform ANOVA test for comparison
#     groups = []
#     for house in houses:
#         house_scores = df_course[df_course['Hogwarts House'] == house][course_name].dropna()
#         if len(house_scores) > 0:
#             groups.append(house_scores.values)

#     if len(groups) < 2:
#         print("Not enough groups to perform ANOVA for comparison")
#         return 1.0

#     f_stat, p_value = f_oneway(*groups)
#     print(f"ANOVA p-value: {p_value:.5f}")

#     return p_value



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
        
        # Calculate homogeneity for all courses using ANOVA
        # Higher p-value means distributions are more similar (more homogeneous)
        numerical_courses = df.select_dtypes(include=['float64'])
        p_values = {}
        f_stats = {}
        
        print("\n" + "="*60)
        print("CALCULATING HOMOGENEITY FOR ALL COURSES")
        print("="*60)
        for course in numerical_courses.columns:
            p_val, f_stat = check_homogeneity(course, df)
            p_values[course] = p_val
            f_stats[course] = f_stat
        
        # Display histograms for all courses
        print("\n" + "="*60)
        print("PLOTTING HISTOGRAMS FOR ALL COURSES")
        print("="*60)
        histogram(df)
        
        # Find the most homogeneous course (highest p-value)
        # Use multiple tiebreakers: variance of means, variance of medians, range of means, F-statistic
        if p_values:
            max_p_value = max(p_values.values())
            
            # Find all courses with the maximum p-value
            courses_with_max_p = [course for course, p_val in p_values.items() if p_val == max_p_value]
            
            # If there are ties, use multiple tiebreakers
            if len(courses_with_max_p) > 1:
                print(f"\nNote: {len(courses_with_max_p)} courses have the same p-value ({max_p_value:.5f})")
                print("Using multiple tiebreakers (lower values = more homogeneous):")
                print("-" * 60)
                
                # Calculate all tiebreaker metrics
                tiebreaker_metrics = {}
                for course in courses_with_max_p:
                    var_means = calculate_variance_of_means(course, df)
                    var_medians = calculate_variance_of_medians(course, df)
                    range_means = calculate_range_of_means(course, df)
                    f_stat = f_stats[course]
                    
                    tiebreaker_metrics[course] = {
                        'var_means': var_means,
                        'var_medians': var_medians,
                        'range_means': range_means,
                        'f_stat': f_stat
                    }
                    
                    print(f"{course:30s}: var_means={var_means:.6f}, var_medians={var_medians:.6f}, "
                          f"range_means={range_means:.6f}, F-stat={f_stat:.6f}")
                
                # Try to break ties using multiple criteria
                # Lower values are better for all metrics
                remaining_courses = courses_with_max_p.copy()
                
                # Try tiebreaker 1: variance of means
                var_means_values = {c: tiebreaker_metrics[c]['var_means'] for c in remaining_courses}
                min_var_means = min(var_means_values.values())
                remaining_courses = [c for c in remaining_courses if var_means_values[c] == min_var_means]
                
                # Try tiebreaker 2: variance of medians
                if len(remaining_courses) > 1:
                    var_medians_values = {c: tiebreaker_metrics[c]['var_medians'] for c in remaining_courses}
                    min_var_medians = min(var_medians_values.values())
                    remaining_courses = [c for c in remaining_courses if var_medians_values[c] == min_var_medians]
                
                # Try tiebreaker 3: range of means
                if len(remaining_courses) > 1:
                    range_means_values = {c: tiebreaker_metrics[c]['range_means'] for c in remaining_courses}
                    min_range_means = min(range_means_values.values())
                    remaining_courses = [c for c in remaining_courses if range_means_values[c] == min_range_means]
                
                # Try tiebreaker 4: F-statistic (lower is better when p-value is same)
                if len(remaining_courses) > 1:
                    f_stat_values = {c: tiebreaker_metrics[c]['f_stat'] for c in remaining_courses}
                    min_f_stat = min(f_stat_values.values())
                    remaining_courses = [c for c in remaining_courses if f_stat_values[c] == min_f_stat]
                
                # If still tied after all tiebreakers, select the first one alphabetically
                # or report all as equally homogeneous
                if len(remaining_courses) > 1:
                    print(f"\nAll {len(remaining_courses)} courses are statistically identical.")
                    print("Selecting the first one alphabetically.")
                    most_homogeneous_course = sorted(remaining_courses)[0]
                else:
                    most_homogeneous_course = remaining_courses[0]
                
                min_variance = tiebreaker_metrics[most_homogeneous_course]['var_means']
                min_var_medians = tiebreaker_metrics[most_homogeneous_course]['var_medians']
                min_range = tiebreaker_metrics[most_homogeneous_course]['range_means']
                min_f_stat = tiebreaker_metrics[most_homogeneous_course]['f_stat']
            else:
                most_homogeneous_course = courses_with_max_p[0]
                min_variance = calculate_variance_of_means(most_homogeneous_course, df)
                min_var_medians = calculate_variance_of_medians(most_homogeneous_course, df)
                min_range = calculate_range_of_means(most_homogeneous_course, df)
                min_f_stat = f_stats[most_homogeneous_course]
            
            print("\n" + "="*60)
            print("RESULTS: HOMOGENEITY ANALYSIS")
            print("="*60)
            print("\nHomogeneity metrics (higher p-value = more homogeneous, lower others = more homogeneous):")
            print("-" * 60)
            # Sort by p-value descending, then by variance of means ascending (lower is better)
            sorted_courses = sorted(p_values.items(), 
                                   key=lambda x: (x[1], 
                                                calculate_variance_of_means(x[0], df),
                                                calculate_variance_of_medians(x[0], df),
                                                calculate_range_of_means(x[0], df),
                                                f_stats[x[0]]), 
                                   reverse=False)
            sorted_courses.reverse()  # Reverse to get descending order
            
            for course, p_val in sorted_courses:
                marker = " <-- MOST HOMOGENEOUS" if course == most_homogeneous_course else ""
                var_means = calculate_variance_of_means(course, df)
                var_medians = calculate_variance_of_medians(course, df)
                range_means = calculate_range_of_means(course, df)
                f_stat = f_stats[course]
                print(f"{course:30s}: p={p_val:.5f}, var_means={var_means:.6f}, "
                      f"var_medians={var_medians:.6f}, range={range_means:.6f}, F={f_stat:.6f}{marker}")
            
            print("\n" + "="*60)
            print(f"ANSWER: {most_homogeneous_course.upper()}")
            print("="*60)
            print(f"The course with the most homogeneous score distribution")
            print(f"between all four houses is: {most_homogeneous_course}")
            print(f"ANOVA p-value: {max_p_value:.5f}")
            if len(courses_with_max_p) > 1:
                print(f"\nTiebreaker metrics:")
                print(f"  Variance of means: {min_variance:.6f}")
                print(f"  Variance of medians: {min_var_medians:.6f}")
                print(f"  Range of means: {min_range:.6f}")
                print(f"  F-statistic: {min_f_stat:.6f}")
                print(f"\nNote: {len(courses_with_max_p)} courses had the same p-value.")
                # Check if we need to report multiple courses as equal
                # Only check if tiebreaker_metrics was created (when len(courses_with_max_p) > 1)
                if 'tiebreaker_metrics' in locals():
                    final_candidates = [c for c in courses_with_max_p 
                                      if tiebreaker_metrics[c]['var_means'] == min_variance
                                      and tiebreaker_metrics[c]['var_medians'] == min_var_medians
                                      and tiebreaker_metrics[c]['range_means'] == min_range
                                      and tiebreaker_metrics[c]['f_stat'] == min_f_stat]
                    if len(final_candidates) > 1:
                        print(f"All {len(final_candidates)} courses are statistically identical:")
                        for c in sorted(final_candidates):
                            print(f"  - {c}")
                        print(f"Selected {most_homogeneous_course} (first alphabetically).")
                    else:
                        print(f"Selected based on tiebreaker metrics.")
                else:
                    print(f"Selected based on tiebreaker metrics.")
            print(f"\n(A higher p-value indicates more similar distributions)")
            print(f"Histogram saved as: histogram_{most_homogeneous_course.replace(' ', '_')}.png")
        else:
            print("\nNo courses found for analysis.")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()


