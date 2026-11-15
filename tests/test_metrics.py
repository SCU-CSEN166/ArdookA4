from ardook_grants.metrics import fit_scores


def test_perfect_predictions():
    y = ["Low", "Moderate", "High", "Low"]
    s = fit_scores(y, y)
    assert s["accuracy"] == 1 and s["f1"] == 1 and s["high_recall"] == 1
    assert s["confusion"] == [[2, 0, 0], [0, 1, 0], [0, 0, 1]]


def test_always_low_misses_every_high():
    y = ["Low", "Low", "Moderate", "High"]
    s = fit_scores(y, ["Low"] * 4)
    assert s["accuracy"] == 0.5 and s["high_recall"] == 0
    lo, hi = map(float, s["f1_95ci"].split("-"))
    assert 0 <= lo <= s["f1"] <= hi <= 1


def test_empty():
    assert fit_scores([], []) == {"n": 0}
