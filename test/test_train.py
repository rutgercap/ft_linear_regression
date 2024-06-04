from src.train import PriceMileagePair, train


def test_values_are_correct_for_consistent_price() -> None:
    pairs = [PriceMileagePair(1, 0), PriceMileagePair(1, 0)]
    thetas = train(pairs, 0.1)
    assert thetas[0] == 0.1
    assert thetas[1] == 0


def test_somethingelse() -> None:
    pairs = [PriceMileagePair(1, 1), PriceMileagePair(1, 1)]
    thetas = train(pairs, 1)
    assert thetas[0] == 1
    assert thetas[1] == 1


def test_values_are_correct_for_zero_mileage() -> None:
    pairs = [PriceMileagePair(0, 1), PriceMileagePair(0, 1)]
    thetas = train(pairs, 0.1)
    assert thetas[0] == 0
    assert thetas[1] == 0


def test_increasing_learning_rate_increases_effect() -> None:
    pairs = [PriceMileagePair(1, 0), PriceMileagePair(1, 0)]
    thetas = train(pairs, 1)
    assert thetas[0] == 1
    assert thetas[1] == 0
