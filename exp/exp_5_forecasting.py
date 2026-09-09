"""
Experiment 5 – Out-of-sample forecasting comparison.

Models
------
ALM-GARCH : unrestricted (5 params)
LM-GARCH  : joint_symmetric (3 params)
HAR-RV    : Corsi (2009) heterogeneous AR on realized variance
GARCH     : standard GARCH(1,1)
GJR-GARCH : asymmetric GARCH with leverage
EGARCH    : exponential GARCH

Setup
-----
Train  : first 70% of each asset's return series
Eval   : last 30%
Re-estimation frequency: every 250 trading days (expanding window)

ALM-GARCH / LM-GARCH forecast at time t:
    sigma_{t+1}^2 = mu + X_t   (last filtered state)
Between re-estimation dates the filter state is carried forward one step
at a time using the current parameter estimates.

Loss
----
QLIKE_t = RV_t / h_t - log(RV_t / h_t) - 1

Saves
-----
results/exp5_qlike_summary.csv
results/exp5_dm_tests.csv
results/exp5_forecasts_{asset}.csv
results/exp5_forecast_comparison.png
"""

import sys
import os
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.alm_garch import estimate_alm_garch
from src.benchmarks import (
    forecast_har_rv,
    forecast_garch_benchmark,
    qlike_loss,
    diebold_mariano_test,
)
from data.data_loader import load_all_assets, load_all_rv

REESTIM_FREQ = 250    # re-estimation cadence (trading days)
TRAIN_FRAC = 0.70     # fraction of data used for initial training
N_STARTS_INIT = 5     # optimizer starts at initial estimation
N_STARTS_ROLL = 3     # optimizer starts for rolling re-estimations
ALM_MODELS = {
    "ALM-GARCH": "unrestricted",
    "LM-GARCH": "joint_symmetric",
}
GARCH_MODELS = ["GARCH", "GJR-GARCH", "EGARCH"]


# ---------------------------------------------------------------------------
# ALM-GARCH one-step state update (pure Python, called between re-estimations)
# ---------------------------------------------------------------------------

def _alm_one_step(r_t: float, X_prev: float, c_prev: float,
                  mu: float, xi_plus: float, xi_minus: float,
                  gamma_plus: float, gamma_minus: float,
                  p: float, tau: float) -> tuple[float, float]:
    """Advance the ALM-GARCH filter by one observation.

    Returns (X_new, c_new) — the state after processing return r_t.
    """
    xi_s = xi_plus if r_t >= 0.0 else xi_minus
    gamma_s = gamma_plus if r_t >= 0.0 else gamma_minus
    rc = 1.0 / (1.0 + tau / max(c_prev, 1e-10))
    rho = rc ** p
    rc_p1 = rc ** (p + 1.0)
    z = r_t * r_t
    x_new = X_prev * rho + xi_s * z
    den = (X_prev / max(c_prev, 1e-10)) * rc_p1 + (xi_s / gamma_s) * z
    c_new = x_new / den if den > 0.0 else c_prev
    c_new = max(c_new, 1e-10)
    return x_new, c_new


# ---------------------------------------------------------------------------
# ALM-GARCH / LM-GARCH rolling forecast
# ---------------------------------------------------------------------------

def alm_garch_rolling_forecast(
    returns: pd.Series,
    train_size: int,
    model: str = "unrestricted",
    p: float = 1.2,
    tau: float = 1.0,
    reestim_freq: int = REESTIM_FREQ,
) -> pd.Series:
    """Expanding-window one-step-ahead ALM-GARCH forecasts.

    Forecast at index t: sigma_{t+1}^2 = mu + X_t, where X_t is the
    filtered state after processing returns[0..t-1] (available before t).

    Between re-estimations the state is updated manually with fixed
    parameters; full re-estimation happens every `reestim_freq` steps.
    """
    r_arr = np.asarray(returns, dtype=float)
    idx = returns.index
    n = len(r_arr)

    forecasts = {}
    params = None
    X_prev = None
    c_prev = None

    for t in range(train_size, n):
        is_reestim = (t == train_size) or ((t - train_size) % reestim_freq == 0)
        if is_reestim:
            n_st = N_STARTS_INIT if t == train_size else N_STARTS_ROLL
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    res = estimate_alm_garch(
                        r_arr[:t], model=model, p=p, tau=tau,
                        n_starts=n_st, seed=42,
                    )
                    params = res.params
                    X_prev = float(res.filtered_X[-1])
                    c_prev = float(res.filtered_c[-1])
                except Exception:
                    # Keep previous params/state if re-estimation fails
                    pass

        if params is None:
            continue

        # One-step-ahead forecast for returns[t]
        forecasts[idx[t]] = float(params.mu + X_prev)

        # Update state with actual return[t]
        X_prev, c_prev = _alm_one_step(
            r_arr[t], X_prev, c_prev,
            params.mu, params.xi_plus, params.xi_minus,
            params.gamma_plus, params.gamma_minus,
            params.p, params.tau,
        )

    return pd.Series(forecasts, name=f"{model}_forecast")


# ---------------------------------------------------------------------------
# Per-asset forecasting pipeline
# ---------------------------------------------------------------------------

def forecast_asset(
    asset: str,
    returns: pd.Series,
    rv: pd.Series,
) -> tuple[pd.DataFrame, dict]:
    """Generate all model forecasts for one asset.

    Returns
    -------
    fcast_df : DataFrame with columns [rv, ALM-GARCH, LM-GARCH, HAR-RV,
                                        GARCH, GJR-GARCH, EGARCH]
              aligned on the evaluation-period index.
    qlike_means : dict {model_name: mean_qlike}
    """
    n = len(returns)
    train_size = int(TRAIN_FRAC * n)
    eval_idx = returns.index[train_size:]

    print(f"  {asset}: N={n}, train={train_size}, eval={n - train_size}")

    # ---- ALM-GARCH / LM-GARCH ----
    all_forecasts = {}
    for label, model_str in ALM_MODELS.items():
        print(f"    {label}...", end=" ", flush=True)
        try:
            fc = alm_garch_rolling_forecast(
                returns, train_size, model=model_str,
            )
            all_forecasts[label] = fc
            print("done")
        except Exception as exc:
            print(f"FAILED ({exc})")

    # ---- HAR-RV ----
    print("    HAR-RV...", end=" ", flush=True)
    try:
        fc_har = forecast_har_rv(rv)
        all_forecasts["HAR-RV"] = fc_har
        print("done")
    except Exception as exc:
        print(f"FAILED ({exc})")

    # ---- GARCH-family ----
    for garch_name in GARCH_MODELS:
        print(f"    {garch_name}...", end=" ", flush=True)
        try:
            fc_g = forecast_garch_benchmark(
                returns, vol_model=garch_name, reestim_freq=REESTIM_FREQ,
            )
            all_forecasts[garch_name] = fc_g
            print("done")
        except Exception as exc:
            print(f"FAILED ({exc})")

    # ---- Align on evaluation period ----
    rv_eval = rv.reindex(eval_idx)
    df = pd.DataFrame({"rv": rv_eval})
    for label, fc in all_forecasts.items():
        df[label] = fc.reindex(eval_idx)

    # Keep only rows where rv > 0 and all model forecasts are finite and > 0
    model_cols = [c for c in df.columns if c != "rv"]
    valid = df["rv"] > 0
    for col in model_cols:
        if col in df.columns:
            valid &= df[col].apply(lambda x: np.isfinite(x) and x > 0)
    df = df.loc[valid].copy()

    return df


# ---------------------------------------------------------------------------
# QLIKE summary and DM tests
# ---------------------------------------------------------------------------

def compute_qlike_summary(
    asset: str,
    df: pd.DataFrame,
) -> pd.DataFrame:
    model_cols = [c for c in df.columns if c != "rv"]
    rows = []
    for col in model_cols:
        if col not in df.columns:
            continue
        losses = qlike_loss(df["rv"].values, df[col].values)
        rows.append({"asset": asset, "model": col, "mean_qlike": float(np.mean(losses))})
    return pd.DataFrame(rows)


def compute_dm_tests(asset: str, df: pd.DataFrame) -> pd.DataFrame:
    """Pairwise DM tests: ALM-GARCH vs each other model."""
    if "ALM-GARCH" not in df.columns:
        return pd.DataFrame()

    benchmark_cols = [c for c in df.columns if c not in ("rv", "ALM-GARCH")]
    rows = []
    alm_loss = qlike_loss(df["rv"].values, df["ALM-GARCH"].values)

    for col in benchmark_cols:
        if col not in df.columns:
            continue
        bm_loss = qlike_loss(df["rv"].values, df[col].values)
        result = diebold_mariano_test(alm_loss, bm_loss)
        rows.append({
            "asset":     asset,
            "model_1":   "ALM-GARCH",
            "model_2":   col,
            "dm_stat":   result["statistic"],
            "p_value":   result["p_value"],
            "mean_diff": result["mean_diff"],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_forecast_comparison(
    qlike_summary: pd.DataFrame,
    path: str,
) -> None:
    """Bar chart of mean QLIKE by model for each asset."""
    assets = qlike_summary["asset"].unique()
    models = qlike_summary["model"].unique()

    fig, axes = plt.subplots(
        nrows=2, ncols=3,
        figsize=(14, 8),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for i, asset in enumerate(assets):
        if i >= len(axes_flat):
            break
        ax = axes_flat[i]
        sub = qlike_summary[qlike_summary["asset"] == asset].set_index("model")
        vals = sub["mean_qlike"].reindex(models).dropna()
        vals.plot(kind="bar", ax=ax, color="steelblue", edgecolor="black", width=0.6)
        ax.set_title(asset, fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("Mean QLIKE")
        ax.tick_params(axis="x", rotation=35, labelsize=7)

    # Hide unused subplots
    for j in range(len(assets), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Out-of-sample QLIKE by Model and Asset", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs("results", exist_ok=True)

    print("=== Experiment 5: Out-of-sample Forecasting ===")
    print("Loading data...")
    assets_returns = load_all_assets()
    assets_rv = load_all_rv()

    all_qlike = []
    all_dm = []

    for asset in assets_returns:
        print(f"\nAsset: {asset}")
        returns = assets_returns[asset]
        rv = assets_rv[asset]

        df = forecast_asset(asset, returns, rv)

        # Save per-asset forecasts
        df.to_csv(f"results/exp5_forecasts_{asset}.csv")
        print(f"  Saved results/exp5_forecasts_{asset}.csv  ({len(df)} eval obs)")

        # QLIKE summary
        qlike_df = compute_qlike_summary(asset, df)
        all_qlike.append(qlike_df)

        # DM tests
        dm_df = compute_dm_tests(asset, df)
        if not dm_df.empty:
            all_dm.append(dm_df)

    # --- Save aggregate results ---
    qlike_summary = pd.concat(all_qlike, ignore_index=True)
    qlike_summary.to_csv("results/exp5_qlike_summary.csv", index=False)
    print("\nSaved results/exp5_qlike_summary.csv")
    print(qlike_summary.to_string(index=False))

    if all_dm:
        dm_summary = pd.concat(all_dm, ignore_index=True)
        dm_summary.to_csv("results/exp5_dm_tests.csv", index=False)
        print("\nSaved results/exp5_dm_tests.csv")
        print(dm_summary.to_string(index=False))

    plot_forecast_comparison(qlike_summary, "results/exp5_forecast_comparison.png")

    print("\nExperiment 5 complete.")


if __name__ == "__main__":
    main()
