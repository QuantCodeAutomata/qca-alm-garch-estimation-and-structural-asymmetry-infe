"""
Bootstrap procedures for LR-test calibration of ALM-GARCH.

Two resampling schemes
----------------------
gaussian      : simulate from fitted restricted model with N(0,1) innovations.
symmetrized   : resample standardized residuals from the null-model fit and
                apply a random sign-flip (Mammen 1993) to enforce symmetry.

p-value formula
---------------
    p = (1 + #{LR_b ≥ lr_observed}) / (B + 1)

Design note
-----------
All bootstrap replications run sequentially in the calling process.  This
avoids spawning loky workers which would force numba to re-JIT the filter
in every child process, causing a severe slowdown.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
from scipy import optimize as sp_optimize

try:
    from alm_garch import (
        ALMGARCHParams, ALMGARCHResult,
        estimate_alm_garch, simulate_alm_garch, lr_test,
        alm_garch_filter, alm_garch_loglik,
    )
except ModuleNotFoundError:
    from src.alm_garch import (
        ALMGARCHParams, ALMGARCHResult,
        estimate_alm_garch, simulate_alm_garch, lr_test,
        alm_garch_filter, alm_garch_loglik,
    )


# ---------------------------------------------------------------------------
# Restriction → model-string mapping
# ---------------------------------------------------------------------------

_NULL_MODEL = {
    "level":  "level_restricted",
    "memory": "memory_restricted",
    "joint":  "joint_symmetric",
}
_RESTRICTION_DF = {"level": 1, "memory": 1, "joint": 2}


# ---------------------------------------------------------------------------
# Fast single-start L-BFGS-B estimator for bootstrap reps
# ---------------------------------------------------------------------------

def _fast_estimate_boot(
    r: np.ndarray,
    model: str,
    p: float,
    tau: float,
    n_starts: int,
    seed: int,
) -> ALMGARCHResult:
    """Light-weight ALM-GARCH estimator for bootstrap replications.

    Uses only L-BFGS-B (no Nelder-Mead) with reduced maxiter for speed.
    This trades some precision for the ~10× speed gain needed in bootstrap
    simulation loops.
    """
    # Import bound/pack helpers from estimate_alm_garch module
    from src.alm_garch import (
        _BOUNDS, _pack_theta, _unpack_theta,
        alm_garch_loglik, alm_garch_filter,
        ALMGARCHParams, ALMGARCHResult,
    )

    r = np.asarray(r, dtype=np.float64)
    sample_var = float(np.var(r))
    log_sv = np.log(max(sample_var, 1e-8))

    # A few structured warm starts; n_starts random starts on top
    structured = [
        _pack_theta(log_sv - 4.0, np.log(0.05), np.log(0.05),
                    np.log(50.0), np.log(50.0), model),
        _pack_theta(log_sv - 3.5, np.log(0.03), np.log(0.10),
                    np.log(20.0), np.log(5.0),  model),
    ]
    rng = np.random.RandomState(seed)
    random_starts = []
    for _ in range(n_starts):
        lm  = rng.uniform(log_sv - 5.0, log_sv + 1.0)
        lxp = rng.uniform(-6.0, 2.0)
        lxm = rng.uniform(-4.0, 3.0)
        lgp = rng.uniform(-1.0, 6.0)
        lgm = rng.uniform(-1.0, 6.0)
        random_starts.append(_pack_theta(lm, lxp, lxm, lgp, lgm, model))

    all_starts = structured + random_starts
    bounds = _BOUNDS[model]

    best_nll  = np.inf
    best_theta = None

    for theta0 in all_starts:
        try:
            res = sp_optimize.minimize(
                alm_garch_loglik, theta0,
                args=(r, model, p, tau),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-9, 'gtol': 1e-6},
            )
            if np.isfinite(res.fun) and res.fun < best_nll:
                best_nll   = res.fun
                best_theta = res.x.copy()
        except Exception:
            pass

    if best_theta is None:
        best_theta = structured[0]

    mu, xi_p, xi_m, gp, gm = _unpack_theta(best_theta, model)
    params = ALMGARCHParams(mu=mu, xi_plus=xi_p, xi_minus=xi_m,
                            gamma_plus=gp, gamma_minus=gm, p=p, tau=tau)
    X_arr, c_arr, sigma2, loglik = alm_garch_filter(r, mu, xi_p, xi_m, gp, gm, p, tau)
    residuals = r / np.sqrt(np.maximum(sigma2, 1e-16))

    return ALMGARCHResult(
        params=params, loglik=loglik, n_obs=len(r),
        filtered_X=X_arr, filtered_c=c_arr,
        filtered_sigma2=sigma2, residuals=residuals,
        converged=(best_nll < 1e9),
    )


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def simulate_null_gaussian(
    null_params: ALMGARCHParams,
    n: int,
    seed: int,
) -> np.ndarray:
    """Simulate from a fitted restricted ALM-GARCH model with Gaussian innovations."""
    return simulate_alm_garch(null_params, n=n, seed=seed)


def simulate_null_symmetrized(
    null_params: ALMGARCHParams,
    residuals: np.ndarray,
    n: int,
    seed: int,
) -> np.ndarray:
    """Symmetrized-residual bootstrap (Mammen 1993) for ALM-GARCH.

    Resamples standardized residuals from the null-model fit and applies
    random Rademacher sign-flips to enforce the null symmetry condition.
    """
    rng = np.random.RandomState(seed)

    std = float(np.std(residuals))
    eta_pool = residuals / max(std, 1e-12)

    idx = rng.randint(0, len(eta_pool), size=n)
    eta = eta_pool[idx] * rng.choice(np.array([-1.0, 1.0]), size=n)

    mu       = null_params.mu
    xi_plus  = null_params.xi_plus
    xi_minus = null_params.xi_minus
    gp       = null_params.gamma_plus
    gm       = null_params.gamma_minus
    p        = null_params.p
    tau      = null_params.tau

    v = float(np.var(eta))
    x_prev = max(v, 1e-8)
    c_prev = max(gp, gm)

    returns = np.empty(n)
    for t in range(n):
        sig2 = max(mu + x_prev, 1e-16)
        r_t  = np.sqrt(sig2) * eta[t]
        returns[t] = r_t

        xi_s    = xi_plus  if r_t >= 0.0 else xi_minus
        gamma_s = gp       if r_t >= 0.0 else gm

        rc    = 1.0 / (1.0 + tau / c_prev)
        rho   = rc ** p
        rc_p1 = rc ** (p + 1.0)
        z     = r_t * r_t

        x_new = x_prev * rho + xi_s * z
        den   = (x_prev / c_prev) * rc_p1 + (xi_s / gamma_s) * z
        c_new = x_new / den if den > 0.0 else c_prev
        c_prev = max(c_new, 1e-10)
        x_prev = x_new

    return returns


# ---------------------------------------------------------------------------
# Main bootstrap routine  (sequential — no joblib/multiprocessing)
# ---------------------------------------------------------------------------

def bootstrap_lr_pvalue(
    returns: np.ndarray,
    restriction: str,
    lr_observed: float,
    B: int = 49,
    scheme: str = "gaussian",
    seed: int = 42,
    n_starts: int = 5,
    n_starts_boot: int = 2,
    p: float = 1.2,
    tau: float = 1.0,
    null_result: Optional[ALMGARCHResult] = None,
    verbose: bool = False,
) -> dict:
    """Compute bootstrap p-value for an ALM-GARCH likelihood-ratio test.

    Runs all replications sequentially in the calling process to prevent
    numba re-JIT in spawned child processes.

    Parameters
    ----------
    returns        : observed return series.
    restriction    : 'level', 'memory', or 'joint'.
    lr_observed    : observed LR statistic (from lr_test on real data).
    B              : number of bootstrap replications (default 49).
    scheme         : 'gaussian' or 'symmetrized'.
    seed           : master RNG seed.
    n_starts       : optimizer starts when fitting the null model to observed
                     data (only used when null_result is None).
    n_starts_boot  : optimizer starts per bootstrap replication (lighter).
    p              : power exponent (fixed).
    tau            : memory-decay parameter (fixed).
    null_result    : pre-fitted restricted model; if provided, the routine
                     skips re-fitting the null model and uses these params.
    verbose        : print a progress dot every 10 reps if True.

    Returns
    -------
    dict with keys:
        p_value, lr_bootstrap_distribution, null_params, converged_fraction
    """
    if restriction not in _NULL_MODEL:
        raise ValueError(f"restriction must be one of {list(_NULL_MODEL)}")
    if scheme not in ("gaussian", "symmetrized"):
        raise ValueError(f"scheme must be 'gaussian' or 'symmetrized'")

    returns = np.asarray(returns, dtype=float)
    n = len(returns)
    null_model_str = _NULL_MODEL[restriction]
    df = _RESTRICTION_DF[restriction]

    # Fit (or reuse) null model on observed data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if null_result is None:
            null_result = estimate_alm_garch(
                returns, model=null_model_str, p=p, tau=tau,
                n_starts=max(n_starts, 5), seed=seed,
            )

    lr_dist = []
    n_failed = 0

    for b in range(B):
        seed_b = seed + b * 97 + 1

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Simulate under null
            if scheme == "gaussian":
                r_b = simulate_null_gaussian(null_result.params, n, seed_b)
            else:
                r_b = simulate_null_symmetrized(
                    null_result.params, null_result.residuals, n, seed_b
                )

            # Fit restricted model (null)
            try:
                res_null_b = _fast_estimate_boot(
                    r_b, null_model_str, p, tau, n_starts_boot, seed_b,
                )
            except Exception:
                n_failed += 1
                continue

            # Fit unrestricted model
            try:
                res_unres_b = _fast_estimate_boot(
                    r_b, "unrestricted", p, tau, n_starts_boot, seed_b + 7,
                )
            except Exception:
                n_failed += 1
                continue

            lr_b, _ = lr_test(res_unres_b.loglik, res_null_b.loglik, df)
            if np.isfinite(lr_b):
                lr_dist.append(float(lr_b))
            else:
                n_failed += 1

        if verbose and (b + 1) % 10 == 0:
            print(f".", end="", flush=True)

    if verbose:
        print()

    lr_dist_arr = np.array(lr_dist)
    converged_fraction = len(lr_dist_arr) / B

    p_value = (1.0 + float(np.sum(lr_dist_arr >= lr_observed))) / (B + 1.0)

    return {
        "p_value":                   float(p_value),
        "lr_bootstrap_distribution": lr_dist_arr,
        "null_params":               null_result.params,
        "converged_fraction":        float(converged_fraction),
        "n_failed":                  n_failed,
    }
