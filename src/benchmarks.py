"""
Benchmark volatility models for comparison with ALM-GARCH.

Models implemented
------------------
HAR-RV   : Heterogeneous Autoregressive Realized Volatility (Corsi 2009).
GARCH    : standard GARCH(1,1) via the arch package.
GJR-GARCH: GARCH with asymmetric leverage (Glosten-Jagannathan-Runkle).
EGARCH   : exponential GARCH (Nelson 1991).
FIGARCH  : fractionally integrated GARCH (Baillie-Bollerslev-Mikkelsen 1996).

All GARCH variants use Gaussian innovations and are estimated via QML.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from arch import arch_model
from scipy import stats


# ---------------------------------------------------------------------------
# HAR-RV
# ---------------------------------------------------------------------------

def _har_regressors(rv: np.ndarray, t: int) -> Optional[np.ndarray]:
    """Return [1, RV_d, RV_w, RV_m] for row t (1-indexed data, t>=22)."""
    if t < 22:
        return None
    rv_d = rv[t - 1]
    rv_w = np.mean(rv[t - 5:t])
    rv_m = np.mean(rv[t - 22:t])
    return np.array([1.0, rv_d, rv_w, rv_m])


def fit_har_rv(rv: pd.Series) -> dict:
    """Fit HAR-RV model by OLS.

    Model: RV_t = β₀ + β_d·RV_{t-1} + β_w·RV^w_{t-1} + β_m·RV^m_{t-1} + ε_t

    where RV^w is the 5-day average and RV^m the 22-day average of lagged RV.

    Parameters
    ----------
    rv : pd.Series
        Realized-variance series (must be non-negative).

    Returns
    -------
    dict with keys:
        beta (ndarray [β₀, β_d, β_w, β_m]),
        rsquared, residuals, fitted_values, ols_result
    """
    rv_arr = np.asarray(rv, dtype=float)
    n = len(rv_arr)

    X, y = [], []
    for t in range(22, n):
        row = _har_regressors(rv_arr, t)
        if row is not None:
            X.append(row)
            y.append(rv_arr[t])

    X = np.array(X)
    y = np.array(y)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ols = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 22})

    fitted = ols.fittedvalues
    resid  = y - fitted

    return {
        "beta":         np.asarray(ols.params),
        "rsquared":     float(ols.rsquared),
        "residuals":    resid,
        "fitted_values": fitted,
        "ols_result":   ols,
    }


def forecast_har_rv(rv: pd.Series, n_ahead: int = 1) -> pd.Series:
    """One-step-ahead HAR-RV forecasts using an expanding estimation window.

    The model is re-fitted at each step on all available data up to t-1
    and used to forecast RV_t.

    Parameters
    ----------
    rv      : pd.Series of realized variance.
    n_ahead : currently only 1 is supported.

    Returns
    -------
    pd.Series of forecasts, indexed like rv (starting from the first
    observation where prediction is possible, i.e., index ≥ 23).
    """
    if n_ahead != 1:
        raise NotImplementedError("Only 1-step-ahead forecasts are implemented.")

    rv_arr = np.asarray(rv, dtype=float)
    n = len(rv_arr)
    idx = rv.index if hasattr(rv, "index") else np.arange(n)

    min_obs = 24   # need at least 22 lags + 2 obs for OLS with 4 params
    forecasts = {}

    for t in range(min_obs, n):
        # Fit on rv[0..t-1]
        rv_sub = rv.iloc[:t] if hasattr(rv, "iloc") else rv[:t]
        try:
            fit = fit_har_rv(rv_sub)
            beta = fit["beta"]
        except Exception:
            continue

        # Predict RV_t using regressors built from rv[0..t-1]
        row = _har_regressors(rv_arr, t)
        if row is None:
            continue
        forecasts[idx[t]] = float(np.dot(beta, row))

    return pd.Series(forecasts, name="har_rv_forecast")


# ---------------------------------------------------------------------------
# GARCH-family benchmarks (arch package)
# ---------------------------------------------------------------------------

_ARCH_VOL_MAP = {
    "GARCH":     dict(vol="Garch",   p=1, q=1),
    "GJR-GARCH": dict(vol="GARCH",   p=1, q=1, o=1),
    "EGARCH":    dict(vol="EGARCH",  p=1, q=1),
    "FIGARCH":   dict(vol="FIGARCH", p=1, q=1),
}


def fit_garch_benchmark(returns: pd.Series, vol_model: str = "GARCH") -> object:
    """Fit a GARCH-family model using the arch package.

    Parameters
    ----------
    returns   : return series (raw, not percentage).
    vol_model : one of 'GARCH', 'GJR-GARCH', 'EGARCH', 'FIGARCH'.

    Returns
    -------
    Fitted ARCHModelResult object from the arch package.
    """
    if vol_model not in _ARCH_VOL_MAP:
        raise ValueError(
            f"vol_model must be one of {list(_ARCH_VOL_MAP)}, got {vol_model!r}"
        )

    r = np.asarray(returns, dtype=float) * 100.0  # arch expects %-returns
    spec = _ARCH_VOL_MAP[vol_model]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if vol_model == "GJR-GARCH":
            am = arch_model(r, vol="Garch", p=1, o=1, q=1, dist="Normal")
        elif vol_model == "FIGARCH":
            try:
                am = arch_model(r, vol="FIGARCH", p=1, q=1, dist="Normal")
            except Exception:
                # Fallback to GARCH if FIGARCH fails
                warnings.warn(
                    "FIGARCH unavailable; falling back to GARCH(1,1).",
                    RuntimeWarning,
                    stacklevel=2,
                )
                am = arch_model(r, vol="Garch", p=1, q=1, dist="Normal")
        else:
            am = arch_model(r, vol=spec["vol"], p=spec["p"], q=spec["q"],
                            dist="Normal")

        result = am.fit(disp="off", show_warning=False)

    return result


def forecast_garch_benchmark(
    returns: pd.Series,
    vol_model: str = "GARCH",
    reestim_freq: int = 250,
) -> pd.Series:
    """Expanding-window one-step-ahead conditional-variance forecasts.

    The model is re-estimated every *reestim_freq* observations; between
    re-estimations the filter is updated with fixed parameters.

    Parameters
    ----------
    returns       : return series.
    vol_model     : one of 'GARCH', 'GJR-GARCH', 'EGARCH', 'FIGARCH'.
    reestim_freq  : number of observations between full re-estimations.

    Returns
    -------
    pd.Series of one-step-ahead variance forecasts (in return-squared units,
    i.e., NOT in percent-squared) aligned to the *returns* index.
    """
    r_arr = np.asarray(returns, dtype=float)
    n = len(r_arr)
    idx = returns.index if hasattr(returns, "index") else np.arange(n)

    min_obs = max(reestim_freq, 100)
    forecasts = {}
    last_result = None

    for t in range(min_obs, n):
        # Re-estimate at the start and then every reestim_freq steps
        if (t == min_obs) or ((t - min_obs) % reestim_freq == 0):
            sub = returns.iloc[:t] if hasattr(returns, "iloc") else pd.Series(r_arr[:t])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    last_result = fit_garch_benchmark(sub, vol_model=vol_model)
                except Exception:
                    pass

        if last_result is None:
            continue

        # One-step-ahead forecast from the fitted model using data up to t
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                fcast = last_result.model.forecast(
                    params=last_result.params,
                    horizon=1,
                    reindex=False,
                )
                # Variance in percent-squared → convert back to return-squared
                h = float(fcast.variance.iloc[-1, 0]) / (100.0 ** 2)
            except Exception:
                # Fallback: use last conditional variance
                cv = last_result.conditional_volatility
                h = float(cv.iloc[-1]) ** 2 / (100.0 ** 2)

        forecasts[idx[t]] = h

    return pd.Series(forecasts, name=f"{vol_model}_h_forecast")


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def qlike_loss(rv_actual: np.ndarray, h_forecast: np.ndarray) -> np.ndarray:
    """QLIKE loss: L_t = RV_t/h_t - log(RV_t/h_t) - 1.

    Parameters
    ----------
    rv_actual  : realized variance, shape (n,).
    h_forecast : model forecast of conditional variance, shape (n,).

    Returns
    -------
    ndarray, shape (n,)   — element-wise QLIKE loss.
    """
    rv = np.asarray(rv_actual,  dtype=float)
    h  = np.asarray(h_forecast, dtype=float)
    ratio = rv / np.maximum(h, 1e-300)
    return ratio - np.log(np.maximum(ratio, 1e-300)) - 1.0


# ---------------------------------------------------------------------------
# Diebold-Mariano test
# ---------------------------------------------------------------------------

def diebold_mariano_test(
    loss1: np.ndarray,
    loss2: np.ndarray,
    max_lag: Optional[int] = None,
) -> dict:
    """Diebold-Mariano test with Newey-West HAC variance.

    H₀: E[L₁_t] = E[L₂_t]  (equal predictive accuracy)
    H₁: E[L₁_t] ≠ E[L₂_t]  (two-sided)

    DM = d̄ / √(V̂_NW / n)

    where d_t = L₁_t - L₂_t and V̂_NW is the Newey-West long-run variance
    estimator with Bartlett kernel.

    Parameters
    ----------
    loss1, loss2 : loss series of equal length.
    max_lag      : HAC bandwidth.  Default: ⌊4·(n/100)^{2/9}⌋ (Andrews rule).

    Returns
    -------
    dict with keys: statistic (float), p_value (float), mean_diff (float)
    """
    d = np.asarray(loss1, dtype=float) - np.asarray(loss2, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 4:
        return {"statistic": np.nan, "p_value": np.nan, "mean_diff": np.nan}

    d_bar = np.mean(d)

    if max_lag is None:
        max_lag = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))

    # Newey-West long-run variance
    d_dm = d - d_bar
    gamma0 = float(np.dot(d_dm, d_dm)) / n
    lrv = gamma0
    for k in range(1, max_lag + 1):
        w = 1.0 - k / (max_lag + 1.0)   # Bartlett weight
        gamma_k = float(np.dot(d_dm[k:], d_dm[:-k])) / n
        lrv += 2.0 * w * gamma_k

    lrv = max(lrv, 1e-300)
    dm_stat = d_bar / np.sqrt(lrv / n)
    p_value = float(2.0 * stats.norm.sf(abs(dm_stat)))

    return {
        "statistic":  float(dm_stat),
        "p_value":    p_value,
        "mean_diff":  float(d_bar),
    }
