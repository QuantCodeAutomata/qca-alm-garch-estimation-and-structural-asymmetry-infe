"""
Foster-Lyapunov stability certificate for ALM-GARCH.

Theory
------
Memory-decay and persistence:
    r(c)   = 1 / (1 + tau/c)
    rho(c) = r(c)^p

Gauss-Hermite nodes (U ~ chi-sq(1) = Z^2, Z ~ N(0,1)):
    E[f(U)] ≈ Σ_i (w_i / √π) · f(z_i^2)

Far-field log-drift:
    Λ_δ(c) = Σ_s q_s · E[log(ρ(c) + ξ_s · U)]      q_+ = q_- = 0.5

Carrier-drift envelope:
    κ̄_s(c)  = γ_s · r(c)^{p+1} / (ξ_s · c)
    m(κ)     = E[κ/(κ + U)]
    S_δ(c)   = Σ_s q_s · [-(c - γ_s) + (c + τ - γ_s) · m(κ̄_s(c))]

Minimax certificate:
    M* = inf_{λ>0} sup_{c ≥ γ_min} {Λ_δ(c) + λ · S_δ(c)}
    certificate_satisfied ⟺ M* < 0
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize
from numpy.polynomial.hermite import hermgauss
from typing import Optional


# ---------------------------------------------------------------------------
# Gauss-Hermite nodes for U ~ chi-sq(1)
# ---------------------------------------------------------------------------

def gauss_hermite_nodes(n: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Return nodes u_i = z_i^2 and normalized weights for E[f(U)], U~chi-sq(1).

    hermgauss uses the weight function exp(-x^2), so the normalization
    factor is 1/sqrt(pi) to convert to unit-mass weights.

    Returns
    -------
    u_nodes : ndarray, shape (n,)   — quadrature nodes z_i^2
    weights : ndarray, shape (n,)   — renormalized weights w_i / sqrt(pi)
    """
    z, w = hermgauss(n)
    return z ** 2, w / np.sqrt(np.pi)


# ---------------------------------------------------------------------------
# Scalar expectations
# ---------------------------------------------------------------------------

def compute_m(kappa: float, u_nodes: np.ndarray, weights: np.ndarray) -> float:
    """Compute E[κ/(κ+U)] where U~chi-sq(1) via Gauss-Hermite quadrature.

    Parameters
    ----------
    kappa   : ratio parameter κ ≥ 0.  At κ=0 returns 0; at κ=∞ returns 1.
    u_nodes : quadrature nodes from gauss_hermite_nodes.
    weights : normalized quadrature weights from gauss_hermite_nodes.
    """
    if not np.isfinite(kappa) or kappa > 1e14:
        return 1.0
    if kappa <= 0.0:
        return 0.0
    return float(np.dot(weights, kappa / (kappa + u_nodes)))


# ---------------------------------------------------------------------------
# Lambda and S on a scalar c
# ---------------------------------------------------------------------------

def compute_Lambda(
    c: float,
    xi_plus: float, xi_minus: float,
    gamma_plus: float, gamma_minus: float,
    p: float, tau: float,
    u_nodes: np.ndarray, weights: np.ndarray,
) -> float:
    """Far-field log-drift Λ_δ(c).

    Λ_δ(c) = 0.5 · E[log(ρ(c) + ξ_+ · U)] + 0.5 · E[log(ρ(c) + ξ_- · U)]
    """
    r_c = 1.0 / (1.0 + tau / c)
    rho_c = r_c ** p
    lam_p = np.dot(weights, np.log(np.maximum(rho_c + xi_plus  * u_nodes, 1e-300)))
    lam_m = np.dot(weights, np.log(np.maximum(rho_c + xi_minus * u_nodes, 1e-300)))
    return 0.5 * (float(lam_p) + float(lam_m))


def compute_S(
    c: float,
    xi_plus: float, xi_minus: float,
    gamma_plus: float, gamma_minus: float,
    p: float, tau: float,
    u_nodes: np.ndarray, weights: np.ndarray,
) -> float:
    """Carrier-drift envelope S_δ(c).

    S_δ(c) = 0.5·[-(c-γ_+) + (c+τ-γ_+)·m(κ̄_+(c))]
           + 0.5·[-(c-γ_-) + (c+τ-γ_-)·m(κ̄_-(c))]

    κ̄_s(c) = γ_s · r(c)^{p+1} / (ξ_s · c)
    """
    r_c = 1.0 / (1.0 + tau / c)
    r_pp1 = r_c ** (p + 1.0)

    def _kappa(gamma_s, xi_s):
        if xi_s <= 0.0:
            return np.inf
        return gamma_s * r_pp1 / (xi_s * c)

    def _s_term(gamma_s, xi_s):
        kappa = _kappa(gamma_s, xi_s)
        m = compute_m(kappa, u_nodes, weights)
        return -(c - gamma_s) + (c + tau - gamma_s) * m

    s_p = _s_term(gamma_plus,  xi_plus)
    s_m = _s_term(gamma_minus, xi_minus)
    return 0.5 * (s_p + s_m)


def compute_envelope(
    c: float, lambda_: float,
    xi_plus: float, xi_minus: float,
    gamma_plus: float, gamma_minus: float,
    p: float, tau: float,
    u_nodes: np.ndarray, weights: np.ndarray,
) -> float:
    """Λ_δ(c) + λ · S_δ(c)."""
    lam = compute_Lambda(c, xi_plus, xi_minus, gamma_plus, gamma_minus,
                         p, tau, u_nodes, weights)
    s   = compute_S(c, xi_plus, xi_minus, gamma_plus, gamma_minus,
                    p, tau, u_nodes, weights)
    return lam + lambda_ * s


# ---------------------------------------------------------------------------
# Vectorized helpers for the grid (avoids Python-level loop overhead)
# ---------------------------------------------------------------------------

def _compute_Lambda_vec(
    c_grid: np.ndarray,
    xi_plus: float, xi_minus: float,
    p: float, tau: float,
    u_nodes: np.ndarray, weights: np.ndarray,
) -> np.ndarray:
    """Vectorized Λ_δ over a c-grid. Shape (n_c,)."""
    r_c = 1.0 / (1.0 + tau / c_grid)          # (n_c,)
    rho_c = r_c ** p                            # (n_c,)
    # outer broadcast: (n_c, n_quad)
    args_p = rho_c[:, None] + xi_plus  * u_nodes[None, :]
    args_m = rho_c[:, None] + xi_minus * u_nodes[None, :]
    np.maximum(args_p, 1e-300, out=args_p)
    np.maximum(args_m, 1e-300, out=args_m)
    lam_p = np.log(args_p) @ weights   # (n_c,)
    lam_m = np.log(args_m) @ weights   # (n_c,)
    return 0.5 * (lam_p + lam_m)


def _compute_S_vec(
    c_grid: np.ndarray,
    xi_plus: float, xi_minus: float,
    gamma_plus: float, gamma_minus: float,
    p: float, tau: float,
    u_nodes: np.ndarray, weights: np.ndarray,
) -> np.ndarray:
    """Vectorized S_δ over a c-grid. Shape (n_c,)."""
    r_c   = 1.0 / (1.0 + tau / c_grid)
    r_pp1 = r_c ** (p + 1.0)

    def _m_vec(gamma_s, xi_s):
        if xi_s <= 0.0:
            return np.ones(len(c_grid))
        kappa = gamma_s * r_pp1 / (xi_s * c_grid)   # (n_c,)
        # E[κ/(κ+U)] = Σ_i w_i · κ_c / (κ_c + u_i)  — vectorized over c and i
        # shape: (n_c, n_quad)
        ratio = kappa[:, None] / (kappa[:, None] + u_nodes[None, :])
        return ratio @ weights   # (n_c,)

    m_p = _m_vec(gamma_plus,  xi_plus)
    m_m = _m_vec(gamma_minus, xi_minus)

    s_p = -(c_grid - gamma_plus)  + (c_grid + tau - gamma_plus)  * m_p
    s_m = -(c_grid - gamma_minus) + (c_grid + tau - gamma_minus) * m_m
    return 0.5 * (s_p + s_m)


# ---------------------------------------------------------------------------
# Main certificate computation
# ---------------------------------------------------------------------------

def compute_stability_certificate(
    xi_plus: float,
    xi_minus: float,
    gamma_plus: float,
    gamma_minus: float,
    p: float = 1.2,
    tau: float = 1.0,
    n_c_grid: int = 500,
    c_max_factor: float = 100.0,
    n_quad: int = 64,
) -> dict:
    """Compute M*, λ*, c_argsup for an ALM-GARCH parameter set.

    M* = inf_{λ>0} sup_{c ≥ γ_min} {Λ_δ(c) + λ · S_δ(c)}

    Parameters
    ----------
    xi_plus, xi_minus       : ARCH scale coefficients.
    gamma_plus, gamma_minus : sign-dependent persistence parameters.
    p                       : power exponent (default 1.2).
    tau                     : memory-decay intercept (default 1.0).
    n_c_grid                : grid resolution over c.
    c_max_factor            : c_max = c_max_factor * tau.
    n_quad                  : Gauss-Hermite quadrature order.

    Returns
    -------
    dict with keys:
        M_star, lambda_star, c_argsup,
        S_grid, Lambda_grid, c_grid,
        certificate_satisfied (bool: M_star < 0)
    """
    u_nodes, weights = gauss_hermite_nodes(n_quad)

    gamma_min = min(gamma_plus, gamma_minus)
    c_lo = max(gamma_min, tau * 1e-4)       # stay away from c=0
    c_hi = c_max_factor * tau

    if c_lo >= c_hi:
        c_lo = max(c_hi * 0.01, tau * 1e-4)

    c_grid = np.linspace(c_lo, c_hi, n_c_grid)

    # Pre-compute Λ and S on the grid (vectorized)
    Lambda_grid = _compute_Lambda_vec(
        c_grid, xi_plus, xi_minus, p, tau, u_nodes, weights
    )
    S_grid = _compute_S_vec(
        c_grid, xi_plus, xi_minus, gamma_plus, gamma_minus,
        p, tau, u_nodes, weights
    )

    # -------------------------------------------------------------------
    # Outer minimization: f(λ) = max_c { Λ(c) + λ · S(c) }
    # Convex in λ → use Brent's method with tight bounds.
    # -------------------------------------------------------------------
    def outer_obj(log_lam: float) -> float:
        lam = np.exp(log_lam)
        return float(np.max(Lambda_grid + lam * S_grid))

    # Coarse scan to bracket the minimum
    log_lam_grid = np.linspace(-8.0, 2.0, 200)
    f_coarse = np.array([outer_obj(ll) for ll in log_lam_grid])
    best_idx = int(np.argmin(f_coarse))
    lo = log_lam_grid[max(best_idx - 1, 0)]
    hi = log_lam_grid[min(best_idx + 1, len(log_lam_grid) - 1)]

    res = optimize.minimize_scalar(
        outer_obj,
        bounds=(lo, hi),
        method="bounded",
        options={"xatol": 1e-10},
    )
    lambda_star = float(np.exp(res.x))

    # -------------------------------------------------------------------
    # Refine c_argsup: local continuous maximization around grid maximum
    # -------------------------------------------------------------------
    envelope_at_lstar = Lambda_grid + lambda_star * S_grid
    idx_max = int(np.argmax(envelope_at_lstar))

    # Continuous refinement in a window around the grid maximum
    c_lo_ref = c_grid[max(idx_max - 5, 0)]
    c_hi_ref = c_grid[min(idx_max + 5, n_c_grid - 1)]

    def neg_envelope(c: float) -> float:
        if c <= 0:
            return 1e15
        lam = compute_Lambda(c, xi_plus, xi_minus, gamma_plus, gamma_minus,
                             p, tau, u_nodes, weights)
        s   = compute_S(c, xi_plus, xi_minus, gamma_plus, gamma_minus,
                        p, tau, u_nodes, weights)
        return -(lam + lambda_star * s)

    try:
        ref = optimize.minimize_scalar(
            neg_envelope,
            bounds=(c_lo_ref, c_hi_ref),
            method="bounded",
            options={"xatol": 1e-8},
        )
        c_argsup = float(ref.x)
        M_star   = float(-ref.fun)
    except Exception:
        c_argsup = float(c_grid[idx_max])
        M_star   = float(envelope_at_lstar[idx_max])

    return {
        "M_star":                 M_star,
        "lambda_star":            lambda_star,
        "c_argsup":               c_argsup,
        "S_grid":                 S_grid,
        "Lambda_grid":            Lambda_grid,
        "c_grid":                 c_grid,
        "certificate_satisfied":  bool(M_star < 0.0),
    }


# ---------------------------------------------------------------------------
# Convenience: tabulate over multiple assets
# ---------------------------------------------------------------------------

def stability_table(params_dict: dict) -> pd.DataFrame:
    """Compute stability diagnostics for all assets.

    Parameters
    ----------
    params_dict : dict
        Mapping {asset_name: ALMGARCHParams | dict}.
        Dict entries must have keys: xi_plus, xi_minus, gamma_plus, gamma_minus.
        Optional keys: p (default 1.2), tau (default 1.0).

    Returns
    -------
    pd.DataFrame with columns:
        asset, M_star, lambda_star, c_argsup, certificate_satisfied
    """
    rows = []
    for asset, par in params_dict.items():
        if hasattr(par, "_asdict"):          # NamedTuple (ALMGARCHParams)
            d = par._asdict()
        elif hasattr(par, "as_dict"):        # dataclass version
            d = par.as_dict()
        else:
            d = dict(par)

        cert = compute_stability_certificate(
            xi_plus=float(d["xi_plus"]),
            xi_minus=float(d["xi_minus"]),
            gamma_plus=float(d["gamma_plus"]),
            gamma_minus=float(d["gamma_minus"]),
            p=float(d.get("p", 1.2)),
            tau=float(d.get("tau", 1.0)),
        )
        rows.append(
            {
                "asset":                 asset,
                "M_star":                cert["M_star"],
                "lambda_star":           cert["lambda_star"],
                "c_argsup":              cert["c_argsup"],
                "certificate_satisfied": cert["certificate_satisfied"],
            }
        )

    return pd.DataFrame(rows).set_index("asset")
