import pandas as pd
import sys

def describe():
    ...



def main():
    try:
        if len(sys.argv) != 2:
            raise Exception("Number of arguments is incorrect")
        dataset = pd.read_csv(sys.argv[1])
        
        print(dataset)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()