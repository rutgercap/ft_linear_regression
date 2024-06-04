from src.estimate_price import estimate_price


def test_estimate_price_works_with_zeroes() -> None:
    estimation = estimate_price(100, 0, 0)

    assert estimation == 0

def test_estimate_price_works_with_simple_values() -> None:
    estimation = estimate_price(100, 1, 0)

    assert estimation == 1

def test_estimate_price_works_with_simple_values() -> None:
    estimation = estimate_price(10, 1, 1)

    assert estimation == 11