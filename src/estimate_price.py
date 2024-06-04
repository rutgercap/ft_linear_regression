import sys

def read_thetas(path: str) -> tuple[float, float]:
    try:
        with open(path, "r") as file:
            lines = file.readlines()
            theta_0 = float(lines[0])
            theta_1 = float(lines[1])
    except FileNotFoundError:
        print("File not found")
        exit(1)
    return (theta_0, theta_1)

def estimate_price(mileage: int, theta_0: float, theta_1: float) -> float:
    return theta_0 + theta_1 * mileage

def main() -> None:
    args = sys.argv[1:]
    if len(args) != 2:
        print("Usage: python3 estimate_price.py <path_to_thetas> <mileage>")
        exit(1)
    path_to_weights = args[0]
    thetas = read_thetas(path_to_weights)
    mileage = int(args[1])  
    estimated = estimate_price(mileage, thetas[0], thetas[1])
    print(estimated)
    exit(0)

if __name__ == "__main__":
    main()