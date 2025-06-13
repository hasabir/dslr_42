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
    plt.show()
    
    

def histogram(df):
    dataset = df.select_dtypes(include=['float64'])
    for course in dataset.columns:
        plot_histogram_for_course(course, df)
        # break


def check_homogeneity(course_name, df):
    df_course = df[['Hogwarts House', course_name]].dropna()
    
    # Get groups for each house (ensure all houses exist in the data)
    houses = df['Hogwarts House'].unique()
    groups = []
    
    for house in houses:
        house_scores = df_course[df_course['Hogwarts House'] == house][course_name]
        groups.append(house_scores)
        
    f_stat, p_value = f_oneway(*groups)
    print(f"{course_name}: p-value = {p_value:.5f}")
    return p_value



def check_homogeneity_by_mean_and_median(course_name, df):
    df_course = df[['Hogwarts House', course_name]].dropna()
    
    # Get groups for each house (ensure all houses exist in the data)
    houses = df['Hogwarts House'].unique()
    groups = []
    
    for house in houses:
        house_scores = df_course[df_course['Hogwarts House'] == house][course_name]
        groups.append(house_scores)
        
    f_stat, p_value = f_oneway(*groups)
    print(f"{course_name}: p-value = {p_value:.5f}")
    return p_value



def main():
    try:
        if len(sys.argv) != 2:
            raise Exception("Number of arguments is incorrect")
        df = pd.read_csv(sys.argv[1])
        # print(calculate_homogeneity(df).sum())
        
        tmp = df.select_dtypes(include=['float64'])
        p_value= {}
        for course in tmp.columns:
            p_value[course] = check_homogeneity(course, df)
        histogram(df)
        test = df.groupby('Hogwarts House')['Arithmancy'].agg(['mean', 'median'])
        # test = 
        print(test)
        # p_value = check_homogeneity('Herbology', df)
        print(p_value)

        

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()


