from dataclasses import dataclass
from typing import Sequence
import numpy as np
import pandas as pd
import sys
import os


@dataclass
class PriceMileagePair:
    mileage: int
    price: int


def read_file(path_to_training_data: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path_to_training_data)
    except FileNotFoundError:
        print("File not found")
        exit(1)
    return df


def csv_to_pairs(df: pd.DataFrame) -> Sequence[PriceMileagePair]:
    if "km" not in df.columns or "price" not in df.columns:
        print(
            "Invalid file contents. Ensure that the file contains columns 'km' and 'price'."
        )
        exit(1)

    try:
        pairs = [PriceMileagePair(row["km"], row["price"]) for _, row in df.iterrows()]
    except KeyError:
        print(
            "Invalid file contents. Ensure that the file contains columns 'km' and 'price'."
        )
        exit(1)
    return pairs


def gradient_descent(
    pairs: Sequence[PriceMileagePair],
    learning_rate: float = 0.0001,
    epochs: int = 100000,
) -> tuple[float, float]:
    mileage = np.array([p.mileage for p in pairs], dtype=np.float64)
    price = np.array([p.price for p in pairs], dtype=np.float64)

    mileage_mean = mileage.mean()
    mileage_std = mileage.std()
    price_mean = price.mean()
    price_std = price.std()

    mileage = (mileage - mileage_mean) / mileage_std
    price = (price - price_mean) / price_std

    theta_0 = 0
    theta_1 = 0
    m = len(pairs)

    for iteration in range(epochs):
        predictions = theta_0 * mileage + theta_1

        errors = predictions - price

        cost = (1 / (2 * m)) * np.sum(errors**2)

        gradient_0 = (1 / m) * np.sum(errors * mileage)
        gradient_1 = (1 / m) * np.sum(errors)

        theta_0 -= learning_rate * gradient_0
        theta_1 -= learning_rate * gradient_1

        if iteration % 1000 == 0:
            print(
                f"Iteration {iteration}: Cost {cost}, theta_0 {theta_0}, theta_1 {theta_1}"
            )

    theta_0 = theta_0 * price_std / mileage_std
    theta_1 = price_mean - (theta_0 * mileage_mean)

    return theta_0, theta_1


def train(pairs: Sequence[PriceMileagePair]) -> tuple[float, float]:
    thetas = gradient_descent(pairs)
    return thetas


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
    pairs = csv_to_pairs(df)
    thetas = train(pairs)
    weights_dir = "weights"
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
    weights_file = os.path.join(weights_dir, "weights.txt")
    with open(weights_file, "w") as f:
        f.write(f"{thetas[0]}\n")
        f.write(f"{thetas[1]}\n")
        print(f"Weights saved to {weights_file}")


if __name__ == "__main__":
    main()
