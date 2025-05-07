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
        'Min': min(args),
        '25%': Statistic.quantile(args, 0.25),
        '50%': Statistic.quantile(args, 0.5),
        '75%': Statistic.quantile(args, 0.75),
        'Max': max(args),
    }
    return f"{stats[key]:.6f}"

def describe(df):
    print(df)
    print('**********************************************************')
    column_types = df.dtypes
    indexes = ["Count", "Mean", "Std", 'Min', '25%', '50%', '75%', 'Max']

    describe = pd.DataFrame()

    df = df.select_dtypes(include=['float64'])
    for statisc in indexes:
        describe_columns = {}
        for column_name in df.columns:
            column = df[column_name].dropna().values.astype(float)
            describe_columns[column_name] = get_statistics(statisc, column)
        new_row = pd.DataFrame(describe_columns, index=[statisc])
        describe = pd.concat([describe, new_row])
    print(describe)



def main():
    try:
        if len(sys.argv) != 2:
            raise Exception("Number of arguments is incorrect")
        df = pd.read_csv(sys.argv[1])
        describe(df)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()