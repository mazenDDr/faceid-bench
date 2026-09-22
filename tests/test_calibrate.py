import numpy as np
import pytest

from faceid_bench.calibrate import (
    Isotonic,
    Naive,
    Platt,
    cllr,
    ece,
    min_cllr,
    pav,
)


def synthetic(n_gen=2000, n_imp=20000, seed=0):
    """Scores whose true LLR is known: genuine ~ N(1, 1), impostor ~ N(-1, 1) -> LLR = 2s."""
    rng = np.random.default_rng(seed)
    s = np.concatenate([rng.normal(1, 1, n_gen), rng.normal(-1, 1, n_imp)])
    y = np.concatenate([np.ones(n_gen, bool), np.zeros(n_imp, bool)])
    return s, y


def test_platt_recovers_true_llr():
    s, y = synthetic()
    m = Platt().fit(s, y)
    assert m.a == pytest.approx(2.0, abs=0.1)
    assert m.b == pytest.approx(0.0, abs=0.1)


def test_uninformative_system_has_cllr_one_and_perfect_zero():
    y = np.array([True, False] * 50)
    assert cllr(np.zeros(100), y) == pytest.approx(1.0)
    assert cllr(np.where(y, 50.0, -50.0), y) == pytest.approx(0.0, abs=1e-12)


def test_calibrated_beats_naive_and_min_cllr_is_a_floor():
    s, y = synthetic()
    s_test, y_test = synthetic(seed=1)
    fitted = cllr(Platt().fit(s, y).llr(s_test), y_test)
    naive = cllr(Naive().llr(np.tanh(s_test / 3)), y_test)
    assert fitted < naive
    assert min_cllr(s_test, y_test) <= fitted + 1e-9


def test_pav_is_monotone_and_preserves_weighted_mean():
    rng = np.random.default_rng(3)
    y, w = rng.random(200), rng.random(200)
    fit = pav(y, w)
    assert np.all(np.diff(fit) >= -1e-12)
    assert (fit * w).sum() == pytest.approx((y * w).sum())


def test_isotonic_is_monotone_and_finite():
    s, y = synthetic()
    m = Isotonic().fit(s, y)
    grid = np.linspace(-6, 6, 500)
    llr = m.llr(grid)
    assert np.all(np.isfinite(llr)) and np.all(np.diff(llr) >= -1e-12)


def test_ece_near_zero_for_true_llr_and_large_for_overconfident():
    s, y = synthetic(20000, 20000)
    assert ece(2 * s, y) < 0.02
    assert ece(20 * s, y) > 0.1


def test_boot_metrics_with_unit_weights_equal_point_metrics():
    from faceid_bench.calibrate import boot_metrics, brier

    rng = np.random.default_rng(5)
    n_ids = 12
    id_gen = rng.integers(0, n_ids, 300)
    id_a, id_b = rng.integers(0, n_ids, 3000), rng.integers(0, n_ids, 3000)
    keep = id_a != id_b
    id_a, id_b = id_a[keep], id_b[keep]
    llr_gen, llr_imp = rng.normal(2, 2, 300), rng.normal(-3, 2, len(id_a))
    out = boot_metrics(llr_gen, id_gen, llr_imp, id_a, id_b, n_ids, np.ones((2, n_ids)))
    llr = np.concatenate([llr_gen, llr_imp])
    y = np.concatenate([np.ones(300, bool), np.zeros(len(llr_imp), bool)])
    assert out["cllr"] == pytest.approx([cllr(llr, y)] * 2)
    assert out["ece"] == pytest.approx([ece(llr, y)] * 2)
    assert out["brier"] == pytest.approx([brier(llr, y)] * 2)


def test_probability_shifts_with_prior():
    from faceid_bench.calibrate import probability

    assert probability(0.0) == pytest.approx(0.5)
    assert probability(0.0, prior=0.9) == pytest.approx(0.9)
    assert probability(np.log(99), prior=0.5) == pytest.approx(0.99)
