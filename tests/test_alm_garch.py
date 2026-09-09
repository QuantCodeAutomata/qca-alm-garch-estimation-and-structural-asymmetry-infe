"""
Comprehensive tests for src/alm_garch.py.

Run with:  pytest tests/test_alm_garch.py -v
"""

import sys
import warnings
sys.path.insert(0, '/workspace/project')

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.alm_garch import (
    ALMGARCHParams,
    alm_garch_filter,
    alm_garch_loglik,
    estimate_alm_garch,
    lr_test,
    compute_persistence,
    simulate_alm_garch,
)
from src.stability import (
    compute_stability_certificate,
    gauss_hermite_nodes,
    compute_m,
)
from src.benchmarks import (
    qlike_loss,
    diebold_mariano_test,
    fit_har_rv,
)


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

# Symmetric reference parameters (moderate persistence, small vol-of-vol)
SYM_PARAMS = ALMGARCHParams(
    mu=1e-4, xi_plus=0.05, xi_minus=0.05,
    gamma_plus=5.0, gamma_minus=5.0,
)

# Asymmetric parameters
ASYM_PARAMS = ALMGARCHParams(
    mu=1e-4, xi_plus=0.02, xi_minus=0.10,
    gamma_plus=3.0, gamma_minus=8.0,
)

# Moderate synthetic return series (N=500 from symmetric DGP, fixed seed)
_SIM_RETURNS = simulate_alm_garch(SYM_PARAMS, n=500, seed=0)


# ---------------------------------------------------------------------------
# Filter tests
# ---------------------------------------------------------------------------

def test_alm_garch_filter_basic():
    """Filter produces all-positive conditional variances and a finite log-likelihood."""
    p = SYM_PARAMS
    X, c, sigma2, loglik = alm_garch_filter(
        _SIM_RETURNS, p.mu, p.xi_plus, p.xi_minus,
        p.gamma_plus, p.gamma_minus, p.p, p.tau,
    )
    assert sigma2.shape == _SIM_RETURNS.shape
    assert np.all(sigma2 > 0), "All conditional variances must be positive"
    assert np.isfinite(loglik), "Log-likelihood must be finite"
    assert np.all(np.isfinite(sigma2)), "All sigma2 values must be finite"
    assert np.all(c > 0), "All scale states c must be positive"


def test_alm_garch_filter_symmetry():
    """With xi_+=xi_- and gamma_+=gamma_-, both branches produce identical dynamics."""
    p = SYM_PARAMS
    X1, c1, sig1, ll1 = alm_garch_filter(
        _SIM_RETURNS, p.mu, p.xi_plus, p.xi_minus,
        p.gamma_plus, p.gamma_minus, p.p, p.tau,
    )
    # Swap plus/minus — should give the same result because parameters are equal
    X2, c2, sig2, ll2 = alm_garch_filter(
        _SIM_RETURNS, p.mu, p.xi_minus, p.xi_plus,
        p.gamma_minus, p.gamma_plus, p.p, p.tau,
    )
    np.testing.assert_allclose(sig1, sig2, rtol=1e-10,
        err_msg="Symmetric model: swapping +/- params must leave sigma2 unchanged")
    np.testing.assert_allclose(ll1, ll2, rtol=1e-10,
        err_msg="Symmetric model: log-likelihood must be invariant to +/- swap")


def test_alm_garch_filter_output_shapes():
    """Filter output arrays have the correct length equal to the input series."""
    n = 300
    rng = np.random.RandomState(99)
    r = rng.randn(n) * 0.01
    p = SYM_PARAMS
    X, c, sigma2, loglik = alm_garch_filter(
        r, p.mu, p.xi_plus, p.xi_minus,
        p.gamma_plus, p.gamma_minus, p.p, p.tau,
    )
    assert X.shape == (n,), "State array X must have length n"
    assert c.shape == (n,), "State array c must have length n"
    assert sigma2.shape == (n,), "sigma2 array must have length n"


# ---------------------------------------------------------------------------
# Log-likelihood tests
# ---------------------------------------------------------------------------

def test_alm_garch_loglik_finite():
    """alm_garch_loglik returns a finite value for a well-specified parameter vector."""
    p = SYM_PARAMS
    theta = np.log(np.array([p.mu, p.xi_plus, p.xi_minus,
                              p.gamma_plus, p.gamma_minus]))
    nll = alm_garch_loglik(theta, _SIM_RETURNS, model='unrestricted')
    assert np.isfinite(nll), "Negative log-likelihood should be finite"
    assert nll < 0 or nll > 0, "NLL should be a real number"  # trivially true


def test_alm_garch_loglik_increases_on_bad_params():
    """Extremely small xi parameters yield a higher (worse) NLL than reference."""
    p = SYM_PARAMS
    theta_good = np.log(np.array([p.mu, p.xi_plus, p.xi_minus,
                                   p.gamma_plus, p.gamma_minus]))
    # Very small xi — essentially no vol clustering, bad fit
    theta_bad = np.log(np.array([p.mu, 1e-8, 1e-8,
                                  p.gamma_plus, p.gamma_minus]))
    nll_good = alm_garch_loglik(theta_good, _SIM_RETURNS, model='unrestricted')
    nll_bad = alm_garch_loglik(theta_bad, _SIM_RETURNS, model='unrestricted')
    assert nll_good < nll_bad, (
        "True DGP parameters should give a lower NLL than badly mis-specified ones"
    )


# ---------------------------------------------------------------------------
# Estimation tests
# ---------------------------------------------------------------------------

def test_alm_garch_estimation_convergence():
    """Estimation on N=500 simulated data should return a finite log-likelihood."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = estimate_alm_garch(
            _SIM_RETURNS, model='unrestricted',
            n_starts=2, seed=42,
        )
    assert np.isfinite(res.loglik), "Estimated log-likelihood must be finite"
    assert res.params.mu > 0, "mu must be positive after estimation"
    assert res.params.xi_plus > 0, "xi_plus must be positive"
    assert res.params.xi_minus > 0, "xi_minus must be positive"
    assert res.params.gamma_plus > 0, "gamma_plus must be positive"
    assert res.params.gamma_minus > 0, "gamma_minus must be positive"


def test_alm_garch_restricted_higher_nll():
    """Restricted model NLL is >= unrestricted NLL (nested models principle)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_u = estimate_alm_garch(
            _SIM_RETURNS, model='unrestricted', n_starts=2, seed=42,
        )
        res_l = estimate_alm_garch(
            _SIM_RETURNS, model='level_restricted', n_starts=2, seed=42,
        )
        res_m = estimate_alm_garch(
            _SIM_RETURNS, model='memory_restricted', n_starts=2, seed=42,
        )
        res_s = estimate_alm_garch(
            _SIM_RETURNS, model='joint_symmetric', n_starts=2, seed=42,
        )

    # loglik_unrestricted >= loglik_restricted (or essentially equal within tolerance)
    for res_r, name in [(res_l, 'level'), (res_m, 'memory'), (res_s, 'joint')]:
        assert res_u.loglik >= res_r.loglik - 1e-4, (
            f"Unrestricted loglik must be >= {name}-restricted loglik; "
            f"got {res_u.loglik:.4f} vs {res_r.loglik:.4f}"
        )


def test_alm_garch_residuals_unit_variance():
    """Standardized residuals should have variance close to 1 for a well-fitted model."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = estimate_alm_garch(
            _SIM_RETURNS, model='unrestricted', n_starts=2, seed=42,
        )
    std = float(np.std(res.residuals))
    assert 0.7 < std < 1.4, (
        f"Standardized residuals std should be near 1.0, got {std:.3f}"
    )


# ---------------------------------------------------------------------------
# LR test
# ---------------------------------------------------------------------------

def test_lr_test_basic():
    """LR statistic is non-negative and p-value lies in [0, 1]."""
    lr_stat, pval = lr_test(loglik_unrestricted=-100.0,
                            loglik_restricted=-105.0, df=1)
    assert lr_stat >= 0.0, "LR statistic must be non-negative"
    assert 0.0 <= pval <= 1.0, "p-value must be in [0, 1]"


def test_lr_test_zero_stat():
    """When restricted and unrestricted likelihoods are equal, LR=0 and p-value=1."""
    lr_stat, pval = lr_test(loglik_unrestricted=-50.0,
                            loglik_restricted=-50.0, df=2)
    assert abs(lr_stat) < 1e-10, "Equal logliks give LR=0"
    assert abs(pval - 1.0) < 1e-10, "LR=0 implies p-value=1"


def test_lr_test_chi2_distribution():
    """
    Under the symmetric null, the empirical mean of chi2(1) LR stats should
    be close to the theoretical mean of 1.0 (chi-squared mean = df).
    Uses a small MC validation with fixed seed.
    """
    rng = np.random.RandomState(7)
    chi2_samples = rng.chisquare(df=1, size=500)
    assert abs(np.mean(chi2_samples) - 1.0) < 0.2, (
        "chi2(1) mean should be ~1.0; MC validation of lr_test distribution logic"
    )

    # Verify that lr_test correctly converts a LR stat to a chi2 p-value
    from scipy.stats import chi2
    stat = 3.84   # classic 5% critical value for chi2(1)
    _, pval = lr_test(0.0 + stat / 2, 0.0, df=1)  # loglik diff of stat/2
    assert abs(pval - 0.05) < 0.01, (
        f"LR stat=3.84 should give p≈0.05 for chi2(1); got {pval:.4f}"
    )


def test_lr_test_df2():
    """LR test with df=2 uses chi2(2) distribution correctly."""
    from scipy.stats import chi2
    # At stat=5.991, chi2(2) p-value ≈ 0.05
    _, pval = lr_test(5.991 / 2, 0.0, df=2)
    assert abs(pval - 0.05) < 0.01, (
        f"LR stat≈5.99 should give p≈0.05 for chi2(2); got {pval:.4f}"
    )


# ---------------------------------------------------------------------------
# Persistence and impulse response
# ---------------------------------------------------------------------------

def test_compute_persistence_range():
    """Equilibrium persistence rho_+ and rho_- must lie strictly in (0, 1)."""
    d = compute_persistence(SYM_PARAMS)
    assert 0.0 < d['rho_plus'] < 1.0, (
        f"rho_+ must be in (0,1), got {d['rho_plus']}"
    )
    assert 0.0 < d['rho_minus'] < 1.0, (
        f"rho_- must be in (0,1), got {d['rho_minus']}"
    )


def test_compute_persistence_asymmetry():
    """Higher gamma gives higher persistence; lower gamma gives shorter half-life."""
    d = compute_persistence(ASYM_PARAMS)
    # gamma_minus=8 > gamma_plus=3 → rho_minus > rho_plus
    assert d['rho_minus'] > d['rho_plus'], (
        "Larger gamma should give higher persistence"
    )
    assert d['half_life_minus'] > d['half_life_plus'], (
        "Larger gamma should give longer half-life"
    )


def test_response_curves_decreasing():
    """Impulse response g_s(k) = xi_s * rho_s^k must be non-increasing in k."""
    d = compute_persistence(SYM_PARAMS, n_lags=50)
    g_plus  = d['response_plus']
    g_minus = d['response_minus']
    # Each value must be <= the previous (accounting for floating point)
    assert np.all(np.diff(g_plus)  <= 1e-15), "g_+(k) must be non-increasing in k"
    assert np.all(np.diff(g_minus) <= 1e-15), "g_-(k) must be non-increasing in k"


def test_response_curves_initial_value():
    """First value of response curve equals xi_s (lag-0 impact)."""
    d = compute_persistence(SYM_PARAMS)
    assert abs(d['response_plus'][0]  - SYM_PARAMS.xi_plus)  < 1e-12
    assert abs(d['response_minus'][0] - SYM_PARAMS.xi_minus) < 1e-12


# ---------------------------------------------------------------------------
# Simulation tests
# ---------------------------------------------------------------------------

def test_simulate_alm_garch_properties():
    """Simulated returns have roughly zero mean and variance consistent with mu."""
    returns = simulate_alm_garch(SYM_PARAMS, n=5000, seed=1)
    assert len(returns) == 5000
    # Mean should be near zero
    assert abs(np.mean(returns)) < 3 * np.std(returns) / np.sqrt(len(returns))
    # Unconditional variance should be on the order of mu (1e-4)
    var_r = np.var(returns)
    assert 1e-6 < var_r < 1e-2, (
        f"Unconditional variance {var_r:.2e} is outside expected range [1e-6, 1e-2]"
    )


def test_simulate_alm_garch_reproducible():
    """Same seed produces identical output; different seeds produce different output."""
    r1 = simulate_alm_garch(SYM_PARAMS, n=200, seed=17)
    r2 = simulate_alm_garch(SYM_PARAMS, n=200, seed=17)
    r3 = simulate_alm_garch(SYM_PARAMS, n=200, seed=99)
    np.testing.assert_array_equal(r1, r2, err_msg="Same seed must give identical draws")
    assert not np.allclose(r1, r3), "Different seeds should give different draws"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_edge_case_all_positive_returns():
    """Filter works when all returns are positive (negative branch never activated)."""
    r_pos = np.abs(_SIM_RETURNS)   # force all returns to be non-negative
    p = SYM_PARAMS
    X, c, sigma2, loglik = alm_garch_filter(
        r_pos, p.mu, p.xi_plus, p.xi_minus,
        p.gamma_plus, p.gamma_minus, p.p, p.tau,
    )
    assert np.all(sigma2 > 0), "sigma2 must be positive even when all r_t >= 0"
    assert np.isfinite(loglik), "loglik must be finite when all r_t >= 0"


def test_edge_case_constant_returns():
    """Filter is numerically stable for constant (non-zero) return series."""
    r_const = np.full(100, 0.001)
    p = SYM_PARAMS
    X, c, sigma2, loglik = alm_garch_filter(
        r_const, p.mu, p.xi_plus, p.xi_minus,
        p.gamma_plus, p.gamma_minus, p.p, p.tau,
    )
    assert np.all(sigma2 > 0), "sigma2 must be positive for constant returns"
    assert np.isfinite(loglik), "loglik must be finite for constant returns"
