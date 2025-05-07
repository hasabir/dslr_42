import pandas as pd
import sys
import numpy as np
from describe import describe
from statistic import Statistic


def histogram(df):
    # print(df)
    # print("*******************************")
    df.set_index('Hogwarts House', inplace=True)
    df = df.select_dtypes(include=['float64'])
    
    hogwarts_houses = ['Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff']

    for house in hogwarts_houses:
        dataset = df.loc[house]
        run_calculation = {}
        for column_name in dataset.columns:
            column = dataset[column_name].fillna(0.0).values.astype(float)
            median = Statistic.median(column)
            dataset = dataset.copy()  # Ensure it's a copy, not a view
            dataset[f"{column_name}_comparison"] = np.where(column > median, '+', '-')
            run_calculation[column_name] = 
            break
            # result = column.apply(lambda x: '+' if x > median else '-')
            # describe_columns[column_name] = Statistic.median(column)
        run_for_eatch_scor = pd.DataFrame(describe_columns, index=['Median'])
        # print(dataset)
        # run_data = dataset
        print('\n****************************\n')
        print(dataset)
        break
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