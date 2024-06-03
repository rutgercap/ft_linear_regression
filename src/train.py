import pandas as pd
import sys
import os


def read_file(path_to_training_data: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path_to_training_data)
    except FileNotFoundError:
        print("File not found")
        exit(1)
    return df


def main():
    args = sys.argv[1:]
    if len(args) != 1:
        print("Usage: python3 train.py <path_to_training_data.csv>")
        exit(1)
    path_to_training_data = args[0]
    if not path_to_training_data.endswith(".csv"):
        print("Invalid file format. Only CSV files are supported.")
        exit(1)
    df = read_file(path_to_training_data)
    weights_dir = "weights"
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
    weights_file = os.path.join(weights_dir, "weights.txt")
    with open(weights_file, "w") as f:
        f.write("theta0\n")
        f.write("theta1\n")



if __name__ == "__main__":
    main()
