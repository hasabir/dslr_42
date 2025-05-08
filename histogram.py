import pandas as pd
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

def calculate_homogeneity(df):
    df.set_index('Hogwarts House', inplace=True)
    df = df.select_dtypes(include=['float64'])
    
    hogwarts_houses = ['Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff']
    homogenenity_table = pd.DataFrame()
    for house in hogwarts_houses:
        dataset = df.loc[house]
        run_results = {}
        homogeneous_features = {}
        for column_name in dataset.columns:
            column = dataset[column_name].fillna(0.0).values.astype(float)
            median = Statistic.median(column)
            signs = np.where(column > median, '+', '-')
            
            n1 = (signs == '+').sum()
            n2 = (signs == '-').sum()
            if n1 + n2 == 0:
                continue

            expected_number_of_runs = 2 * n1 * n2 / (n1 + n2) + 1
            variance = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / ((n1 + n2) ** 2 * (n1 + n2 - 1))
            z = 1.96
            lower_bound = expected_number_of_runs - z * np.sqrt(variance)
            upper_bound = expected_number_of_runs + z * np.sqrt(variance)
            
            
            runs = count_runs(signs)  
            run_results[column_name] = runs
            
            homogeneous_features[column_name] = True if runs >= lower_bound and runs <= upper_bound else False

        new_row = pd.DataFrame(homogeneous_features, index=[f"homogeneous_{house}"])
        homogenenity_table = pd.concat([homogenenity_table, new_row])
    return homogenenity_table

import matplotlib.pyplot as plt

def plot_histogram_for_course(course_name, data):
    plt.figure(figsize=(10, 6))
    houses = data['Hogwarts House'].unique()
    
    for house in houses:
        house_data = data[data['Hogwarts House'] == house][course_name]
        plt.hist(house_data, bins=20, alpha=0.5, label=house)
        # print(house_data, data['Hogwarts House'] == house)
        # break
    plt.title(f'Score Distribution for {course_name}')
    plt.xlabel('Scores')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

def histogram(df):
    dataset = df.select_dtypes(include=['float64'])
    for course in dataset.columns:
        plot_histogram_for_course(course, df)
        # break

# def histogram(df):
#     plt.show()




def main():
    try:
        if len(sys.argv) != 2:
            raise Exception("Number of arguments is incorrect")
        df = pd.read_csv(sys.argv[1])
        # print(calculate_homogeneity(df).sum())
        
        histogram(df)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()