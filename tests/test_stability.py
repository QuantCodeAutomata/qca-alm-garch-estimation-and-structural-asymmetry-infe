"""
Tests for src/stability.py — Foster-Lyapunov stability certificate.

Run with:  pytest tests/test_stability.py -v
"""

import sys
sys.path.insert(0, '/workspace/project')

import numpy as np
import pytest

from src.stability import (
    gauss_hermite_nodes,
    compute_m,
    compute_Lambda,
    compute_S,
    compute_stability_certificate,
)
from src.alm_garch import ALMGARCHParams


# ---------------------------------------------------------------------------
# Gauss-Hermite quadrature tests
# ---------------------------------------------------------------------------

def test_gauss_hermite_normalization():
    """Quadrature weights must sum to 1 (E[1] = 1 under any distribution)."""
    u_nodes, weights = gauss_hermite_nodes(n=64)
    total = float(np.sum(weights))
    assert abs(total - 1.0) < 1e-10, (
        f"Weights must sum to 1.0 for unit-mass quadrature; got {total:.12f}"
    )


def test_gauss_hermite_nodes_positive():
    """Quadrature nodes u_i = z_i^2 must all be non-negative."""
    u_nodes, weights = gauss_hermite_nodes(n=32)
    assert np.all(u_nodes >= 0.0), "Nodes z_i^2 must be non-negative"


def test_gauss_hermite_mean_quadrature_node():
    """
    The nodes are u_i = z_i^2 where z_i are hermgauss nodes.  The
    corresponding quadrature integrates against the density (1/√π)·exp(−x²),
    so E[x²] = ∫ x² · (1/√π)·exp(−x²) dx = 1/2.
    """
    u_nodes, weights = gauss_hermite_nodes(n=64)
    mean_u = float(np.dot(weights, u_nodes))
    assert abs(mean_u - 0.5) < 1e-6, (
        f"Quadrature E[z²] should equal 0.5; got {mean_u:.8f}"
    )


def test_gauss_hermite_second_moment():
    """
    E[z⁴] under the (1/√π)·exp(−x²) weight = ∫ x⁴·(1/√π)·exp(−x²)dx = 3/4.
    The node variance is E[u²] − (E[u])² = 3/4 − 1/4 = 1/2.
    """
    u_nodes, weights = gauss_hermite_nodes(n=64)
    mean_u  = float(np.dot(weights, u_nodes))
    mean_u2 = float(np.dot(weights, u_nodes ** 2))
    assert abs(mean_u2 - 0.75) < 1e-6, (
        f"Quadrature E[z⁴] should equal 0.75; got {mean_u2:.8f}"
    )
    var_u = mean_u2 - mean_u ** 2
    assert abs(var_u - 0.5) < 1e-6, (
        f"Quadrature Var[z²] should equal 0.5; got {var_u:.8f}"
    )


def test_gauss_hermite_increasing_order():
    """Higher quadrature order gives the same (or better) approximation of E[z²]=0.5."""
    u16, w16 = gauss_hermite_nodes(n=16)
    u64, w64 = gauss_hermite_nodes(n=64)
    err16 = abs(float(np.dot(w16, u16)) - 0.5)
    err64 = abs(float(np.dot(w64, u64)) - 0.5)
    assert err64 <= err16 + 1e-14, (
        "Higher order quadrature should not increase error for smooth integrals"
    )


# ---------------------------------------------------------------------------
# compute_m tests
# ---------------------------------------------------------------------------

def test_compute_m_bounds():
    """E[kappa/(kappa+U)] should lie strictly in (0, 1) for finite positive kappa."""
    u_nodes, weights = gauss_hermite_nodes(n=64)
    for kappa in [0.01, 0.1, 1.0, 5.0, 50.0, 500.0]:
        m = compute_m(kappa, u_nodes, weights)
        assert 0.0 < m < 1.0, (
            f"compute_m({kappa}) = {m} is not in (0, 1)"
        )


def test_compute_m_monotone_increasing():
    """E[kappa/(kappa+U)] is increasing in kappa."""
    u_nodes, weights = gauss_hermite_nodes(n=64)
    kappas = [0.01, 0.1, 1.0, 10.0, 100.0]
    ms = [compute_m(k, u_nodes, weights) for k in kappas]
    for i in range(len(ms) - 1):
        assert ms[i] < ms[i + 1], (
            f"compute_m should increase with kappa; "
            f"m({kappas[i]})={ms[i]:.4f} >= m({kappas[i+1]})={ms[i+1]:.4f}"
        )


def test_compute_m_limits():
    """At kappa→0 the expectation→0; at kappa→∞ it→1."""
    u_nodes, weights = gauss_hermite_nodes(n=64)
    # Near zero
    m_small = compute_m(1e-12, u_nodes, weights)
    assert m_small < 0.01, f"m(1e-12) should be near 0, got {m_small}"
    # Large kappa: compute_m returns 1.0 for kappa > 1e14
    m_large = compute_m(1e15, u_nodes, weights)
    assert abs(m_large - 1.0) < 1e-10, f"m(1e15) should be 1.0, got {m_large}"


def test_compute_m_kappa_zero():
    """At kappa=0 exactly, compute_m returns 0.0."""
    u_nodes, weights = gauss_hermite_nodes(n=64)
    m = compute_m(0.0, u_nodes, weights)
    assert m == 0.0, f"compute_m(0) must be 0.0, got {m}"


# ---------------------------------------------------------------------------
# Lambda and S tests
# ---------------------------------------------------------------------------

def test_compute_lambda_negative_for_stable():
    """For a stable (sub-critical) model, Lambda should be negative at large c."""
    u_nodes, weights = gauss_hermite_nodes(n=64)
    # Symmetric, moderate persistence
    lam = compute_Lambda(
        c=10.0, xi_plus=0.05, xi_minus=0.05,
        gamma_plus=5.0, gamma_minus=5.0,
        p=1.2, tau=1.0,
        u_nodes=u_nodes, weights=weights,
    )
    # Lambda is the log-drift; for a stable model it should be negative
    assert lam < 0.0, (
        f"Lambda should be negative for a stable model at c=10; got {lam:.6f}"
    )


def test_compute_s_finite():
    """compute_S should return a finite value for reasonable parameters."""
    u_nodes, weights = gauss_hermite_nodes(n=64)
    s = compute_S(
        c=5.0, xi_plus=0.05, xi_minus=0.08,
        gamma_plus=4.0, gamma_minus=7.0,
        p=1.2, tau=1.0,
        u_nodes=u_nodes, weights=weights,
    )
    assert np.isfinite(s), f"compute_S should return a finite value; got {s}"


# ---------------------------------------------------------------------------
# compute_stability_certificate tests
# ---------------------------------------------------------------------------

def test_stability_certificate_stable_model():
    """A clearly stable (low-xi) model should have certificate_satisfied=True."""
    cert = compute_stability_certificate(
        xi_plus=0.01, xi_minus=0.01,
        gamma_plus=5.0, gamma_minus=5.0,
        p=1.2, tau=1.0,
    )
    assert cert["certificate_satisfied"] is True, (
        "Stable model (small xi) should satisfy M* < 0"
    )
    assert np.isfinite(cert["M_star"]), "M_star should be finite"
    assert np.isfinite(cert["lambda_star"]), "lambda_star should be finite"
    assert cert["lambda_star"] > 0.0, "lambda_star should be positive"


def test_stability_certificate_sign():
    """For a highly persistent (near-unit-root) model, M* is borderline or positive."""
    # Very high xi → possibly unstable
    cert = compute_stability_certificate(
        xi_plus=0.5, xi_minus=0.5,
        gamma_plus=100.0, gamma_minus=100.0,
        p=1.2, tau=1.0,
    )
    # Just check that the function runs and returns a dict with the right keys
    required_keys = {"M_star", "lambda_star", "c_argsup",
                     "certificate_satisfied", "S_grid", "Lambda_grid", "c_grid"}
    assert required_keys.issubset(cert.keys()), (
        f"Missing keys: {required_keys - cert.keys()}"
    )
    assert isinstance(cert["certificate_satisfied"], bool)


def test_stability_certificate_return_types():
    """All scalar outputs should be Python floats / booleans."""
    cert = compute_stability_certificate(
        xi_plus=0.05, xi_minus=0.05,
        gamma_plus=5.0, gamma_minus=5.0,
    )
    assert isinstance(cert["M_star"], float)
    assert isinstance(cert["lambda_star"], float)
    assert isinstance(cert["c_argsup"], float)
    assert isinstance(cert["certificate_satisfied"], bool)


def test_stability_certificate_grid_shapes():
    """Lambda_grid, S_grid and c_grid should all have the same length."""
    n_c = 100
    cert = compute_stability_certificate(
        xi_plus=0.05, xi_minus=0.08,
        gamma_plus=4.0, gamma_minus=7.0,
        n_c_grid=n_c,
    )
    assert cert["c_grid"].shape[0] == n_c
    assert cert["Lambda_grid"].shape[0] == n_c
    assert cert["S_grid"].shape[0] == n_c


def test_stability_certificate_asymmetric():
    """Certificate is computed successfully for asymmetric parameters."""
    cert = compute_stability_certificate(
        xi_plus=0.01, xi_minus=0.10,
        gamma_plus=3.0, gamma_minus=8.0,
        p=1.2, tau=1.0,
    )
    assert np.isfinite(cert["M_star"]), "M_star must be finite for asymmetric params"
    assert cert["lambda_star"] > 0.0, "lambda_star must be positive"


def test_stability_certificate_c_argsup_in_range():
    """The argmax c* should lie within the search grid."""
    cert = compute_stability_certificate(
        xi_plus=0.05, xi_minus=0.05,
        gamma_plus=5.0, gamma_minus=5.0,
        n_c_grid=500,
    )
    c_lo = cert["c_grid"][0]
    c_hi = cert["c_grid"][-1]
    # Allow a tiny tolerance around the grid boundary due to continuous refinement
    assert c_lo - 0.1 <= cert["c_argsup"] <= c_hi + 0.1, (
        f"c_argsup={cert['c_argsup']:.3f} is outside grid [{c_lo:.3f}, {c_hi:.3f}]"
    )
