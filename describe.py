import pandas as pd
import sys
import numpy as np

from typing import Any, Callable, Dict, List



class Statistic():
    @staticmethod
    def mean(args):
        return sum(args)/len(args)

    @staticmethod
    def median(args):
        n = int(len(args)/2)
        list = sorted(args)
        return list[n] if len(args) % 2 != 0 else list[n] + (list[n] - 1)

    @staticmethod
    def quantile(args, percentage):
        quantiles = {
            0.25    : sorted(args)[int(len(args)/4)],
            0.5     : sorted(args)[int(len(args)/2)],
            0.75    : sorted(args)[int(3 * len(args)/4)]
        }
        return quantiles[percentage]

    @staticmethod
    def var(args):
        stock = []
        for arg in args:
            stock.append((arg - Statistic.mean(args)) ** 2)
        return sum(stock)/len(args)

    @staticmethod
    def std(args):
        return Statistic.var(args)**(1/2)


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

def describe(file_name):
    df = pd.read_csv(file_name)
    column_types = df.dtypes
    indexes = ["Count", "Mean", "Std", 'Min', '25%', '50%', '75%', 'Max']

    describe = pd.DataFrame()

    
    for statisc in indexes:
        describe_columns = {}
        for column_name in df.columns:
            if df[column_name].dtypes.name == 'float64':
                column = df[column_name].dropna().values.astype(float)
                describe_columns[column_name] = get_statistics(statisc, column)
        new_row = pd.DataFrame(describe_columns, index=[statisc])
        describe = pd.concat([describe, new_row])
    print(describe)



def main():
    try:
        if len(sys.argv) != 2:
            raise Exception("Number of arguments is incorrect")
        describe(sys.argv[1])
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()