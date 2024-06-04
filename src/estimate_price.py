def read_thetas(path: str) -> tuple[float, float]:
    try:
        with open(path, "r") as file:
            lines = file.readlines()
            theta_0 = float(lines[0])
            theta_1 = float(lines[1])
    except FileNotFoundError:
        print("No weights found. Defaulting to 0, 0")
        return (0, 0)
    except ValueError:
        print("Expected theta format:\n<float>\n<float>")
        exit(1)
    return (theta_0, theta_1)


def estimate_price(mileage: int, theta_0: float, theta_1: float) -> float:
    return theta_0 + theta_1 * mileage


def main() -> None:
    path_to_weights = "weights/weights.txt"
    thetas = read_thetas(path_to_weights)
    while True:
        try:
            mileage = int(input("Enter mileage: "))
            estimated = estimate_price(mileage, thetas[0], thetas[1])
            print(estimated)
        except ValueError:
            print("Please enter a valid integer")


if __name__ == "__main__":
    main()
