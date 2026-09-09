"""
ALM-GARCH model implementation.

Volatility recursion
--------------------
  sigma_n^2  = mu + X_{n-1}
  r_n        = sigma_n * epsilon_n,   epsilon_n ~ N(0,1)

Memory-decay function
  r(c) = 1 / (1 + tau/c),   rho(c) = r(c)^p      [tau=1, p=1.2 by default]

Sign-dependent state update  (s = '+' if r_n >= 0, else '-')
  X_n  = X_{n-1} * rho(c_{n-1}) + xi_s * r_n^2
  c_n  = [X_{n-1} * rho + xi_s * z] /
         [(X_{n-1}/c_{n-1}) * r(c)^{p+1} + (xi_s/gamma_s) * z]
         with  z = r_n^2

Initialization
  X_0 = sample variance of returns
  c_0 = max(gamma_+, gamma_-)

Estimation
  Gaussian QML:  sum_n  -0.5 * [log(2pi) + log(sigma_n^2) + r_n^2/sigma_n^2]
  Parameters optimised in log-space for positivity constraints.
"""

import numpy as np
import pandas as pd
from scipy import optimize, stats
from typing import NamedTuple, Optional

# ---------------------------------------------------------------------------
# Try to use numba; fall back silently to pure-Python loops if unavailable.
# ---------------------------------------------------------------------------
try:
    from numba import njit as _njit
    _NUMBA_OK = True
except Exception:
    def _njit(func=None, **kw):          # type: ignore[misc]
        return func if func is not None else (lambda f: f)
    _NUMBA_OK = False


# ---------------------------------------------------------------------------
# Named tuples
# ---------------------------------------------------------------------------

class ALMGARCHParams(NamedTuple):
    mu: float
    xi_plus: float
    xi_minus: float
    gamma_plus: float
    gamma_minus: float
    p: float = 1.2
    tau: float = 1.0


class ALMGARCHResult(NamedTuple):
    params: ALMGARCHParams
    loglik: float
    n_obs: int
    filtered_X: np.ndarray
    filtered_c: np.ndarray
    filtered_sigma2: np.ndarray
    residuals: np.ndarray          # standardized: r_n / sigma_n
    converged: bool


# ---------------------------------------------------------------------------
# Core recursive filter — numba JIT with pure-Python fallback
# ---------------------------------------------------------------------------

@_njit(cache=True)
def _filter_core(
    returns: np.ndarray,
    X0: float,
    c0: float,
    mu: float,
    xi_plus: float,
    xi_minus: float,
    gamma_plus: float,
    gamma_minus: float,
    p: float,
    tau: float,
) -> tuple:
    """
    Numba-compiled ALM-GARCH filter.

    Returns (X, c_arr, sigma2, loglik) where each array has length N = len(returns).
    X[i]      = X_{i+1}  (state after observation i)
    c_arr[i]  = c_{i+1}
    sigma2[i] = mu + X_{i} = sigma_{i+1}^2  ... wait no.

    Actually:
      sigma2[i] = mu + X_prev  where X_prev = X_i at start of step i
    So sigma2[i] is the conditional variance for observation i.
    X[i] is the updated state after processing observation i.
    """
    LOG_2PI = 1.8378770664093453   # log(2 * pi)
    n = len(returns)

    X_arr = np.empty(n)
    c_arr = np.empty(n)
    sigma2 = np.empty(n)

    x_prev = X0
    c_prev = c0
    loglik = 0.0

    for i in range(n):
        r = returns[i]

        # Conditional variance
        sig2 = mu + x_prev
        if sig2 <= 0.0:
            # Ill-specified region — penalise heavily
            return X_arr, c_arr, sigma2, -1.0e10
        sigma2[i] = sig2

        # Gaussian QML log-likelihood contribution
        loglik += -0.5 * (LOG_2PI + np.log(sig2) + r * r / sig2)

        # Sign-dependent parameters
        if r >= 0.0:
            xi_s = xi_plus
            gamma_s = gamma_plus
        else:
            xi_s = xi_minus
            gamma_s = gamma_minus

        # Memory-decay function and persistence
        rc = 1.0 / (1.0 + tau / c_prev)          # r(c_{n-1})
        rho = rc ** p                              # rho(c_{n-1}) = r(c)^p
        rc_p1 = rc ** (p + 1.0)                   # r(c)^{p+1}

        # Squared return (innovation)
        z = r * r

        # State update: X_n = X_{n-1} * rho + xi_s * z
        x_new = x_prev * rho + xi_s * z

        # c update:  c_n = X_n / denominator
        den = (x_prev / c_prev) * rc_p1 + (xi_s / gamma_s) * z
        if den > 0.0:
            c_new = x_new / den
        elif x_new == 0.0:
            # Both numerator and denominator are zero — keep previous c
            c_new = c_prev
        else:
            c_new = c_prev

        # Floor to avoid division-by-zero on next iteration
        if c_new < 1.0e-10:
            c_new = 1.0e-10

        X_arr[i] = x_new
        c_arr[i] = c_new
        x_prev = x_new
        c_prev = c_new

    return X_arr, c_arr, sigma2, loglik


def _filter_python(
    returns: np.ndarray,
    X0: float,
    c0: float,
    mu: float,
    xi_plus: float,
    xi_minus: float,
    gamma_plus: float,
    gamma_minus: float,
    p: float,
    tau: float,
) -> tuple:
    """Pure-Python fallback filter (same logic as _filter_core)."""
    LOG_2PI = np.log(2.0 * np.pi)
    n = len(returns)

    X_arr = np.empty(n)
    c_arr = np.empty(n)
    sigma2 = np.empty(n)

    x_prev = X0
    c_prev = c0
    loglik = 0.0

    for i in range(n):
        r = returns[i]
        sig2 = mu + x_prev
        if sig2 <= 0.0:
            return X_arr, c_arr, sigma2, -1.0e10
        sigma2[i] = sig2
        loglik += -0.5 * (LOG_2PI + np.log(sig2) + r * r / sig2)

        xi_s = xi_plus if r >= 0.0 else xi_minus
        gamma_s = gamma_plus if r >= 0.0 else gamma_minus

        rc = 1.0 / (1.0 + tau / c_prev)
        rho = rc ** p
        rc_p1 = rc ** (p + 1.0)
        z = r * r

        x_new = x_prev * rho + xi_s * z
        den = (x_prev / c_prev) * rc_p1 + (xi_s / gamma_s) * z
        if den > 0.0:
            c_new = x_new / den
        elif x_new == 0.0:
            c_new = c_prev
        else:
            c_new = c_prev

        c_new = max(c_new, 1.0e-10)
        X_arr[i] = x_new
        c_arr[i] = c_new
        x_prev = x_new
        c_prev = c_new

    return X_arr, c_arr, sigma2, loglik


# ---------------------------------------------------------------------------
# Public filter API
# ---------------------------------------------------------------------------

def alm_garch_filter(
    returns: np.ndarray,
    mu: float,
    xi_plus: float,
    xi_minus: float,
    gamma_plus: float,
    gamma_minus: float,
    p: float = 1.2,
    tau: float = 1.0,
) -> tuple:
    """
    Run the ALM-GARCH recursive filter.

    Parameters
    ----------
    returns : array-like, shape (N,)
        Daily log returns.
    mu, xi_plus, xi_minus, gamma_plus, gamma_minus : float
        Model parameters (all must be strictly positive).
    p : float
        Persistence exponent (default 1.2).
    tau : float
        Memory-decay intercept (default 1.0).

    Returns
    -------
    X : ndarray, shape (N,)
        Filtered state X_1 … X_N.
    c : ndarray, shape (N,)
        Filtered scale c_1 … c_N.
    sigma2 : ndarray, shape (N,)
        Conditional variances sigma_1^2 … sigma_N^2.
    loglik : float
        Gaussian QML log-likelihood.
    """
    r = np.asarray(returns, dtype=np.float64)

    # Initial state
    v = np.var(r)
    X0 = float(v) if np.isfinite(v) and v > 0.0 else 1.0e-4
    c0 = float(max(gamma_plus, gamma_minus))

    try:
        X_arr, c_arr, sigma2, loglik = _filter_core(
            r, X0, c0, float(mu), float(xi_plus), float(xi_minus),
            float(gamma_plus), float(gamma_minus), float(p), float(tau),
        )
    except Exception:
        X_arr, c_arr, sigma2, loglik = _filter_python(
            r, X0, c0, float(mu), float(xi_plus), float(xi_minus),
            float(gamma_plus), float(gamma_minus), float(p), float(tau),
        )

    return X_arr, c_arr, sigma2, loglik


# ---------------------------------------------------------------------------
# Parameter packing / unpacking (log-space ↔ positive-space)
# ---------------------------------------------------------------------------

def _unpack_theta(theta: np.ndarray, model: str):
    """Map log-space theta → (mu, xi_plus, xi_minus, gamma_plus, gamma_minus)."""
    e = np.exp(theta)
    if model == 'unrestricted':
        return e[0], e[1], e[2], e[3], e[4]
    elif model == 'level_restricted':          # xi_+ = xi_-
        return e[0], e[1], e[1], e[2], e[3]
    elif model == 'memory_restricted':         # gamma_+ = gamma_-
        return e[0], e[1], e[2], e[3], e[3]
    elif model == 'joint_symmetric':           # xi_+ = xi_-, gamma_+ = gamma_-
        return e[0], e[1], e[1], e[2], e[2]
    else:
        raise ValueError(f"Unknown model: '{model}'")


def _pack_theta(
    log_mu: float, log_xi_p: float, log_xi_m: float,
    log_gp: float, log_gm: float, model: str,
) -> np.ndarray:
    """Pack log-space values → theta vector appropriate for the given model."""
    if model == 'unrestricted':
        return np.array([log_mu, log_xi_p, log_xi_m, log_gp, log_gm])
    elif model == 'level_restricted':
        return np.array([log_mu, log_xi_p, log_gp, log_gm])
    elif model == 'memory_restricted':
        return np.array([log_mu, log_xi_p, log_xi_m, log_gp])
    elif model == 'joint_symmetric':
        return np.array([log_mu, log_xi_p, log_gp])
    else:
        raise ValueError(f"Unknown model: '{model}'")


_BOUNDS = {
    'unrestricted':     [(-20., 2.), (-10., 6.), (-10., 6.), (-2., 8.), (-2., 8.)],
    'level_restricted': [(-20., 2.), (-10., 6.), (-2., 8.), (-2., 8.)],
    'memory_restricted':[(-20., 2.), (-10., 6.), (-10., 6.), (-2., 8.)],
    'joint_symmetric':  [(-20., 2.), (-10., 6.), (-2., 8.)],
}


# ---------------------------------------------------------------------------
# Negative log-likelihood (for scipy.optimize.minimize)
# ---------------------------------------------------------------------------

def alm_garch_loglik(
    theta: np.ndarray,
    returns: np.ndarray,
    model: str = 'unrestricted',
    p: float = 1.2,
    tau: float = 1.0,
) -> float:
    """
    Gaussian QML negative log-likelihood for ALM-GARCH.

    Parameters
    ----------
    theta : ndarray
        Log-space parameter vector (dimension depends on *model*).
    returns : ndarray
        Daily log returns.
    model : str
        One of 'unrestricted', 'level_restricted', 'memory_restricted',
        'joint_symmetric'.
    p, tau : float
        Fixed shape / memory-decay parameters.

    Returns
    -------
    float
        Negative log-likelihood (minimised by scipy optimisers).
    """
    try:
        mu, xi_p, xi_m, gp, gm = _unpack_theta(theta, model)
    except Exception:
        return 1.0e10

    if min(mu, xi_p, xi_m, gp, gm) <= 0.0:
        return 1.0e10

    _, _, _, loglik = alm_garch_filter(returns, mu, xi_p, xi_m, gp, gm, p, tau)
    if not np.isfinite(loglik):
        return 1.0e10
    return -loglik


# ---------------------------------------------------------------------------
# Multi-start estimation
# ---------------------------------------------------------------------------

def estimate_alm_garch(
    returns: np.ndarray,
    model: str = 'unrestricted',
    p: float = 1.2,
    tau: float = 1.0,
    n_starts: int = 10,
    seed: int = 42,
) -> ALMGARCHResult:
    """
    Estimate ALM-GARCH by Gaussian QML with multi-start optimisation.

    Parameters
    ----------
    returns : array-like
        Daily log returns.
    model : str
        'unrestricted'    — 5 free parameters (xi_+, xi_-, gamma_+, gamma_-  distinct)
        'level_restricted'— 4 free parameters (xi_+ = xi_-)
        'memory_restricted'—4 free parameters (gamma_+ = gamma_-)
        'joint_symmetric' — 3 free parameters (xi_+ = xi_-, gamma_+ = gamma_-)
    p, tau : float
        Fixed shape parameters.
    n_starts : int
        Number of random starting points (in addition to 3 structured ones).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    ALMGARCHResult
    """
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]

    sample_var = float(np.var(r))
    if not np.isfinite(sample_var) or sample_var <= 0.0:
        sample_var = 1.0e-4

    log_sv = np.log(sample_var)

    # ------------------------------------------------------------------
    # 3 structured starting points (diverse parameter regimes)
    # ------------------------------------------------------------------
    structured = [
        # Moderate persistence, symmetric
        _pack_theta(log_sv - 4.0, np.log(0.05), np.log(0.05),
                    np.log(10.0), np.log(10.0), model),
        # High persistence (large gamma), symmetric
        _pack_theta(log_sv - 5.5, np.log(0.02), np.log(0.02),
                    np.log(50.0), np.log(50.0), model),
        # Asymmetric: stronger/slower response to negative shocks
        _pack_theta(log_sv - 3.5, np.log(0.03), np.log(0.10),
                    np.log(20.0), np.log(5.0),  model),
    ]

    # ------------------------------------------------------------------
    # n_starts random starting points
    # ------------------------------------------------------------------
    rng = np.random.RandomState(seed)
    random_starts = []
    for _ in range(n_starts):
        lm  = rng.uniform(log_sv - 5.0, log_sv + 1.0)
        lxp = rng.uniform(-6.0, 3.0)
        lxm = rng.uniform(-6.0, 3.0)
        lgp = rng.uniform(-1.0, 6.5)
        lgm = rng.uniform(-1.0, 6.5)
        random_starts.append(_pack_theta(lm, lxp, lxm, lgp, lgm, model))

    all_starts = structured + random_starts
    bounds     = _BOUNDS[model]

    best_nll  = np.inf
    best_opt  = None
    best_theta = None

    for theta0 in all_starts:
        # Primary: L-BFGS-B (gradient-based, fast)
        try:
            res = optimize.minimize(
                alm_garch_loglik, theta0,
                args=(r, model, p, tau),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 600, 'ftol': 1e-11, 'gtol': 1e-7},
            )
            if np.isfinite(res.fun) and res.fun < best_nll:
                best_nll   = res.fun
                best_opt   = res
                best_theta = res.x.copy()
        except Exception:
            pass

        # Secondary: Nelder-Mead from same start if L-BFGS-B didn't improve
        try:
            res_nm = optimize.minimize(
                alm_garch_loglik, theta0,
                args=(r, model, p, tau),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True},
            )
            if np.isfinite(res_nm.fun) and res_nm.fun < best_nll:
                best_nll   = res_nm.fun
                best_opt   = res_nm
                best_theta = res_nm.x.copy()
        except Exception:
            pass

    # Final polishing pass from best theta with tight tolerances
    if best_theta is not None:
        try:
            res_pol = optimize.minimize(
                alm_garch_loglik, best_theta,
                args=(r, model, p, tau),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-9},
            )
            if np.isfinite(res_pol.fun) and res_pol.fun < best_nll:
                best_nll   = res_pol.fun
                best_opt   = res_pol
                best_theta = res_pol.x.copy()
        except Exception:
            pass

    if best_theta is None:
        # Absolute fallback
        best_theta = structured[0]
        best_opt   = None

    mu, xi_p, xi_m, gp, gm = _unpack_theta(best_theta, model)
    params = ALMGARCHParams(
        mu=mu, xi_plus=xi_p, xi_minus=xi_m,
        gamma_plus=gp, gamma_minus=gm, p=p, tau=tau,
    )

    X_arr, c_arr, sigma2, loglik = alm_garch_filter(r, mu, xi_p, xi_m, gp, gm, p, tau)
    residuals = r / np.sqrt(np.maximum(sigma2, 1e-16))

    converged = bool(best_opt.success) if best_opt is not None and hasattr(best_opt, 'success') else False

    return ALMGARCHResult(
        params=params,
        loglik=loglik,
        n_obs=len(r),
        filtered_X=X_arr,
        filtered_c=c_arr,
        filtered_sigma2=sigma2,
        residuals=residuals,
        converged=converged,
    )


# ---------------------------------------------------------------------------
# Likelihood-ratio test
# ---------------------------------------------------------------------------

def lr_test(
    loglik_unrestricted: float,
    loglik_restricted: float,
    df: int,
) -> tuple:
    """
    Compute the likelihood-ratio statistic and asymptotic chi-squared p-value.

    Parameters
    ----------
    loglik_unrestricted, loglik_restricted : float
        Maximised log-likelihoods.
    df : int
        Degrees of freedom (difference in number of free parameters).

    Returns
    -------
    (lr_stat, pval) : (float, float)
    """
    lr_stat = 2.0 * (loglik_unrestricted - loglik_restricted)
    pval    = float(1.0 - stats.chi2.cdf(lr_stat, df=df))
    return float(lr_stat), pval


# ---------------------------------------------------------------------------
# Persistence and response curves
# ---------------------------------------------------------------------------

def compute_persistence(params: ALMGARCHParams, n_lags: int = 100) -> dict:
    """
    Compute sign-dependent persistence and impulse-response curves.

    The equilibrium persistence for the '+'/'-' branch is
        rho_s = r(gamma_s)^p   where r(gamma_s) = 1 / (1 + tau/gamma_s).

    The response of the conditional variance to a unit squared-return shock
    at lag k (assuming c stays near its equilibrium gamma_s) is
        g_s(k) = xi_s * rho_s^k.

    Parameters
    ----------
    params : ALMGARCHParams
    n_lags : int
        Number of lags for response curves.

    Returns
    -------
    dict with keys:
        'rho_plus', 'rho_minus'          — scalar persistence values in (0,1)
        'half_life_plus', 'half_life_minus' — half-lives in periods
        'response_plus', 'response_minus'   — ndarrays of shape (n_lags,)
    """
    tau = params.tau
    p   = params.p

    def _rho(gamma):
        rc = 1.0 / (1.0 + tau / gamma)
        return rc ** p

    rho_plus  = _rho(params.gamma_plus)
    rho_minus = _rho(params.gamma_minus)

    hl_plus  = np.log(0.5) / np.log(rho_plus)  if rho_plus  > 0.0 else np.inf
    hl_minus = np.log(0.5) / np.log(rho_minus) if rho_minus > 0.0 else np.inf

    k = np.arange(n_lags, dtype=float)
    g_plus  = params.xi_plus  * rho_plus  ** k
    g_minus = params.xi_minus * rho_minus ** k

    return {
        'rho_plus':        rho_plus,
        'rho_minus':       rho_minus,
        'half_life_plus':  float(hl_plus),
        'half_life_minus': float(hl_minus),
        'response_plus':   g_plus,
        'response_minus':  g_minus,
    }


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate_alm_garch(
    params: ALMGARCHParams,
    n: int,
    seed: int = 42,
) -> np.ndarray:
    """
    Simulate N observations from an ALM-GARCH process.

    A burn-in of 500 periods is used to initialise the state and discarded.

    Parameters
    ----------
    params : ALMGARCHParams
    n : int
        Number of observations to return.
    seed : int
        Random seed.

    Returns
    -------
    ndarray, shape (n,)
        Simulated log returns.
    """
    rng = np.random.RandomState(seed)

    mu        = params.mu
    xi_plus   = params.xi_plus
    xi_minus  = params.xi_minus
    gp        = params.gamma_plus
    gm        = params.gamma_minus
    p         = params.p
    tau       = params.tau

    burn_in = 500
    n_total = n + burn_in

    r_out  = np.empty(n_total)
    x_prev = mu                         # reasonable initial state
    c_prev = max(gp, gm)

    for i in range(n_total):
        sig2 = mu + x_prev
        sig2 = max(sig2, 1.0e-16)
        eps  = rng.standard_normal()
        r_i  = np.sqrt(sig2) * eps
        r_out[i] = r_i

        xi_s    = xi_plus  if r_i >= 0.0 else xi_minus
        gamma_s = gp       if r_i >= 0.0 else gm

        rc    = 1.0 / (1.0 + tau / c_prev)
        rho   = rc ** p
        rc_p1 = rc ** (p + 1.0)
        z     = r_i * r_i

        x_new = x_prev * rho + xi_s * z
        den   = (x_prev / c_prev) * rc_p1 + (xi_s / gamma_s) * z
        c_new = x_new / den if den > 0.0 else c_prev
        c_new = max(c_new, 1.0e-10)

        x_prev = x_new
        c_prev = c_new

    return r_out[burn_in:]
