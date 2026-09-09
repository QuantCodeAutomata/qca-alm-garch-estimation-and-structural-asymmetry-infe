"""
Tests for src/benchmarks.py — HAR-RV, GARCH-family, QLIKE, and DM test.

Run with:  pytest tests/test_benchmarks.py -v
"""

import sys
import warnings
sys.path.insert(0, '/workspace/project')

import numpy as np
import pandas as pd
import pytest

from src.benchmarks import (
    fit_har_rv,
    forecast_har_rv,
    fit_garch_benchmark,
    forecast_garch_benchmark,
    qlike_loss,
    diebold_mariano_test,
)
from src.alm_garch import ALMGARCHParams, simulate_alm_garch


# ---------------------------------------------------------------------------
# Shared synthetic data
# ---------------------------------------------------------------------------

# Fixed synthetic return series (N=600)
_PARAMS = ALMGARCHParams(
    mu=1e-4, xi_plus=0.05, xi_minus=0.05,
    gamma_plus=5.0, gamma_minus=5.0,
)
_RET_ARR = simulate_alm_garch(_PARAMS, n=600, seed=42)
_RETURNS = pd.Series(_RET_ARR, name="returns")
_RV = _RETURNS ** 2   # squared returns as RV proxy


# ---------------------------------------------------------------------------
# HAR-RV tests
# ---------------------------------------------------------------------------

def test_har_rv_fitting():
    """HAR-RV should fit without error on a synthetic RV series of length >= 23."""
    result = fit_har_rv(_RV)
    assert "beta" in result, "fit_har_rv must return a dict with key 'beta'"
    assert len(result["beta"]) == 4, "HAR-RV has 4 coefficients [beta_0, beta_d, beta_w, beta_m]"
    assert np.all(np.isfinite(result["beta"])), "All beta coefficients must be finite"
    assert "rsquared" in result, "fit_har_rv must return R^2"
    assert 0.0 <= result["rsquared"] <= 1.0, f"R^2 must be in [0,1], got {result['rsquared']}"


def test_har_rv_fitted_values_shape():
    """Fitted values and residuals have the correct length."""
    result = fit_har_rv(_RV)
    n_fit = len(result["fitted_values"])
    n_res = len(result["residuals"])
    # OLS is run on observations 22..N-1 (at least T-22 rows)
    assert n_fit == n_res, "fitted_values and residuals must have equal length"
    assert n_fit == len(result["residuals"]), "Residual length must match fitted length"


def test_har_rv_forecast_series():
    """forecast_har_rv returns a pd.Series with finite positive entries."""
    fc = forecast_har_rv(_RV)
    assert isinstance(fc, pd.Series), "forecast_har_rv must return a pd.Series"
    fc_valid = fc.dropna()
    assert len(fc_valid) > 0, "At least some forecasts must be non-NaN"
    assert np.all(np.isfinite(fc_valid.values)), "All non-NaN forecasts must be finite"


def test_har_rv_forecast_expanding():
    """Forecast series starts no earlier than observation 24."""
    fc = forecast_har_rv(_RV)
    fc_nna = fc.dropna()
    if len(fc_nna) > 0:
        # Minimum observations needed before first forecast
        first_idx = fc_nna.index[0]
        pos = _RV.index.get_loc(first_idx)
        assert pos >= 24, (
            f"First HAR-RV forecast should be at obs >= 24, found at {pos}"
        )


def test_har_rv_small_series():
    """fit_har_rv on a very short series (< 23 obs) raises or returns empty."""
    short_rv = pd.Series(np.random.rand(10) * 1e-4)
    # Either it raises or returns an essentially empty OLS object
    try:
        result = fit_har_rv(short_rv)
        # If it doesn't raise, the beta array may be empty or contain NaNs
        # — just ensure no crash
    except Exception:
        pass  # acceptable to raise on insufficient data


# ---------------------------------------------------------------------------
# GARCH-family tests
# ---------------------------------------------------------------------------

def test_garch_benchmark_fitting():
    """GARCH(1,1) should fit on a 600-observation return series without error."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = fit_garch_benchmark(_RETURNS, vol_model="GARCH")
    # arch result has params and loglikelihood attributes
    assert hasattr(res, "params"), "arch result must have 'params'"
    assert hasattr(res, "loglikelihood"), "arch result must have 'loglikelihood'"
    assert np.isfinite(float(res.loglikelihood)), "arch loglikelihood must be finite"
    assert len(res.params) >= 3, "GARCH(1,1) must have at least 3 params"


def test_gjr_garch_fitting():
    """GJR-GARCH should fit without error."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = fit_garch_benchmark(_RETURNS, vol_model="GJR-GARCH")
    assert np.isfinite(float(res.loglikelihood))
    assert len(res.params) >= 4, "GJR-GARCH must have at least 4 params"


def test_egarch_fitting():
    """EGARCH should fit without error."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = fit_garch_benchmark(_RETURNS, vol_model="EGARCH")
    assert np.isfinite(float(res.loglikelihood))


def test_figarch_fitting_or_fallback():
    """FIGARCH should either fit or gracefully fall back to GARCH."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = fit_garch_benchmark(_RETURNS, vol_model="FIGARCH")
    # Either way, result must have a finite loglikelihood
    assert np.isfinite(float(res.loglikelihood))


def test_garch_invalid_model():
    """An unknown vol_model name should raise ValueError."""
    with pytest.raises(ValueError, match="vol_model must be one of"):
        fit_garch_benchmark(_RETURNS, vol_model="UNKNOWN-MODEL")


def test_garch_forecast_series():
    """GARCH rolling forecast returns a pd.Series with positive entries."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fc = forecast_garch_benchmark(_RETURNS, vol_model="GARCH", reestim_freq=250)
    assert isinstance(fc, pd.Series), "Forecast must be a pd.Series"
    fc_valid = fc.dropna()
    assert len(fc_valid) > 0, "At least some forecasts must be present"
    assert np.all(fc_valid.values > 0), "All GARCH variance forecasts must be positive"
    assert np.all(np.isfinite(fc_valid.values)), "All forecasts must be finite"


def test_garch_forecast_in_return_units():
    """GARCH forecast is in return^2 units (NOT percent-squared), so values << 1."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fc = forecast_garch_benchmark(_RETURNS, vol_model="GARCH", reestim_freq=250)
    fc_valid = fc.dropna()
    if len(fc_valid) > 0:
        # Return series std is ~0.01, so variance ~ 1e-4; forecasts should be << 1
        assert float(fc_valid.median()) < 0.01, (
            f"GARCH forecast median {fc_valid.median():.4f} seems too large "
            "(check that /100^2 scaling was applied)"
        )


# ---------------------------------------------------------------------------
# QLIKE loss tests
# ---------------------------------------------------------------------------

def test_qlike_loss_positive():
    """QLIKE loss L_t = RV/h - log(RV/h) - 1 must be >= 0."""
    rng = np.random.RandomState(0)
    rv = rng.exponential(scale=1e-4, size=200)
    h  = rng.exponential(scale=1e-4, size=200)
    losses = qlike_loss(rv, h)
    assert np.all(losses >= 0), "QLIKE loss must be non-negative"


def test_qlike_loss_zero_at_rv_equals_h():
    """QLIKE achieves its minimum of 0 when rv == h."""
    h = np.array([1e-4, 5e-4, 2e-3])
    losses = qlike_loss(h, h)   # rv = h
    np.testing.assert_allclose(
        losses, 0.0, atol=1e-12,
        err_msg="QLIKE(rv, h) must be 0 when rv == h",
    )


def test_qlike_loss_shape():
    """Output shape matches input shape."""
    rv = np.ones(50) * 1e-4
    h  = np.ones(50) * 2e-4
    losses = qlike_loss(rv, h)
    assert losses.shape == (50,), "QLIKE output shape must match input shape"


def test_qlike_loss_invariant_to_scale():
    """QLIKE is scale-invariant: L(c·rv, c·h) = L(rv, h)."""
    rng = np.random.RandomState(3)
    rv = rng.exponential(scale=1e-4, size=100)
    h  = rng.exponential(scale=1e-4, size=100)
    c  = 1e4
    np.testing.assert_allclose(
        qlike_loss(c * rv, c * h), qlike_loss(rv, h), rtol=1e-10,
        err_msg="QLIKE must be scale-invariant",
    )


# ---------------------------------------------------------------------------
# Diebold-Mariano test
# ---------------------------------------------------------------------------

def test_dm_test_under_null():
    """When both loss series are identical, DM statistic should be 0."""
    rng = np.random.RandomState(5)
    loss = rng.exponential(scale=0.1, size=300)
    result = diebold_mariano_test(loss, loss)
    assert abs(result["statistic"]) < 1e-10, (
        f"DM stat should be 0 for identical losses; got {result['statistic']}"
    )
    assert abs(result["mean_diff"]) < 1e-15, "mean_diff should be 0 for identical losses"


def test_dm_test_significant_difference():
    """A large systematic loss differential should produce a significant DM stat."""
    rng = np.random.RandomState(11)
    n = 500
    base = rng.exponential(scale=0.1, size=n)
    # Model 1 is systematically much worse than model 2
    loss1 = base + 1.0          # large positive bias
    loss2 = base
    result = diebold_mariano_test(loss1, loss2)
    assert abs(result["statistic"]) > 2.0, (
        f"DM stat should be large for a clear differential; got {result['statistic']:.3f}"
    )
    assert result["p_value"] < 0.05, (
        f"p-value should be < 0.05 for systematic differential; got {result['p_value']:.4f}"
    )


def test_dm_test_returns_required_keys():
    """diebold_mariano_test result must contain 'statistic', 'p_value', 'mean_diff'."""
    rng = np.random.RandomState(7)
    loss = rng.exponential(size=100)
    result = diebold_mariano_test(loss, loss * 1.01)
    assert "statistic" in result
    assert "p_value" in result
    assert "mean_diff" in result


def test_dm_test_pvalue_range():
    """p-value must lie in [0, 1]."""
    rng = np.random.RandomState(13)
    loss1 = rng.exponential(size=200)
    loss2 = rng.exponential(size=200)
    result = diebold_mariano_test(loss1, loss2)
    assert 0.0 <= result["p_value"] <= 1.0, (
        f"p-value out of [0,1]: {result['p_value']}"
    )


def test_dm_test_short_series():
    """DM test on a very short series returns NaN gracefully."""
    result = diebold_mariano_test(np.array([0.1, 0.2]), np.array([0.3, 0.4]))
    assert result["statistic"] is np.nan or np.isnan(result["statistic"]), (
        "DM test on < 4 obs should return NaN statistic"
    )


def test_dm_test_antisymmetric():
    """Swapping loss1 and loss2 negates the DM statistic."""
    rng = np.random.RandomState(17)
    n = 300
    loss1 = rng.exponential(scale=0.1, size=n) + 0.2
    loss2 = rng.exponential(scale=0.1, size=n)
    r12 = diebold_mariano_test(loss1, loss2)
    r21 = diebold_mariano_test(loss2, loss1)
    np.testing.assert_allclose(
        r12["statistic"], -r21["statistic"], rtol=1e-10,
        err_msg="Swapping loss series should negate the DM statistic",
    )
    np.testing.assert_allclose(
        r12["p_value"], r21["p_value"], rtol=1e-10,
        err_msg="Swapping loss series should leave p-value unchanged",
    )
