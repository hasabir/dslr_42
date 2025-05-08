import pandas as pd
import sys
import numpy as np
from describe import describe
from statistic import Statistic



def count_runs(signs):
    runs = 1
    for i in range(1, len(signs)):
        if signs[i] != signs[i-1]:
            runs += 1
    return runs

def histogram(df):
    # print(df)
    # print("*******************************")
    df.set_index('Hogwarts House', inplace=True)
    df = df.select_dtypes(include=['float64'])
    
    hogwarts_houses = ['Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff']

    for house in hogwarts_houses:
        dataset = df.loc[house]
        run_results = {}
        for column_name in dataset.columns:
            column = dataset[column_name].fillna(0.0).values.astype(float)
            median = Statistic.median(column)
            dataset = dataset.copy()
            signs = np.where(column > median, '+', '-')
            # run_calculation[column_name] = (dataset[f"{column_name}_comparison"] == '+').sum()
            runs = count_runs(signs)  # Count how many sign changes
            run_results[column_name] = runs

        run_for_eatch_scor = pd.DataFrame(run_results, index=['runs'])
        print(f"*****************{house}*****************************")
        print(run_for_eatch_scor)
        print('\n****************************\n')
        # print(dataset)
        # break
    # print(df)
    # Gryffindor = df.


def main():
    try:
        if len(sys.argv) != 2:
            raise Exception("Number of arguments is incorrect")
        df = pd.read_csv(sys.argv[1])
        histogram(df)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()