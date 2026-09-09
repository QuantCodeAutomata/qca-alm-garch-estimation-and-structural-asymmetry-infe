"""
Experiment 1: ALM-GARCH Estimation
===================================
Estimates four model variants for each of the six assets, computes LR tests
with asymptotic and bootstrap (Gaussian + symmetrized) p-values, and saves
parameter estimates, LR tables, persistence summaries, and response-curve plots.

Outputs (all in results/)
-------------------------
  exp1_parameters.csv     — parameter estimates for all models × assets
  exp1_lr_tests.csv       — LR statistics and p-values
  exp1_persistence.csv    — xi+, xi-, rho+, rho-, gamma+, gamma-
  exp1_response_curves.png — g_s(k) for k = 0 … 250, all assets
"""

import sys
sys.path.insert(0, '/workspace/project')
sys.path.insert(0, '/workspace/project/src')

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.alm_garch import (
    ALMGARCHParams, ALMGARCHResult,
    estimate_alm_garch, lr_test, compute_persistence,
    simulate_alm_garch, alm_garch_filter,
)
from src.bootstrap import bootstrap_lr_pvalue
from data.data_loader import load_all_assets, load_all_rv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
RESULTS_DIR = os.path.abspath(RESULTS_DIR)
os.makedirs(RESULTS_DIR, exist_ok=True)

MODELS = ['unrestricted', 'level_restricted', 'memory_restricted', 'joint_symmetric']

# restriction key -> (null model string, LR name, degrees of freedom)
RESTRICTIONS = [
    ('level',  'LR_lev', 1),
    ('memory', 'LR_mem', 1),
    ('joint',  'LR_sym', 2),
]

# Bootstrap config: paper uses B=999; reduced for computational feasibility.
# Sequential execution (no loky) avoids numba re-JIT overhead in workers.
B               = 49   # bootstrap replications per test
B_SYMM          = 19   # fewer for symmetrized bootstrap (slower sim)
N_STARTS_MAIN   = 5    # optimizer starts for main estimation
N_STARTS_BOOT   = 2    # lighter starts inside each bootstrap rep


# ---------------------------------------------------------------------------
# Response curve: spec formula g_s(k) = xi_s * [gamma_s/(gamma_s + k*tau)]^p
# ---------------------------------------------------------------------------

def response_curve(xi_s: float, gamma_s: float, p: float, tau: float,
                   k_max: int = 250) -> np.ndarray:
    k = np.arange(k_max + 1, dtype=float)
    return xi_s * (gamma_s / (gamma_s + k * tau)) ** p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Experiment 1: ALM-GARCH Estimation")
    print("=" * 70)

    print("\nLoading asset data ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assets = load_all_assets()

    asset_names = list(assets.keys())
    print(f"  Assets: {asset_names}\n")

    params_rows = []
    lr_rows = []
    persistence_rows = []

    # Response-curve figure (2 rows × 3 cols)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes_flat = axes.flatten()

    for plot_idx, (asset_name, returns_series) in enumerate(assets.items()):
        r = returns_series.values.astype(float)
        print(f"{'─'*70}")
        print(f"  {asset_name}   N = {len(r)}")
        print(f"{'─'*70}")

        # ── Estimate all four models ──────────────────────────────────────
        results = {}
        for model in MODELS:
            print(f"    estimating [{model}] ...", end=" ", flush=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = estimate_alm_garch(r, model=model, p=1.2, tau=1.0,
                                         n_starts=N_STARTS_MAIN, seed=42)
            results[model] = res
            print(f"loglik = {res.loglik:.4f}  converged = {res.converged}")

            prm = res.params
            params_rows.append({
                'asset':       asset_name,
                'model':       model,
                'mu':          prm.mu,
                'xi_plus':     prm.xi_plus,
                'xi_minus':    prm.xi_minus,
                'gamma_plus':  prm.gamma_plus,
                'gamma_minus': prm.gamma_minus,
                'p':           prm.p,
                'tau':         prm.tau,
                'loglik':      res.loglik,
                'n_obs':       res.n_obs,
                'converged':   res.converged,
            })

        unres = results['unrestricted']

        # ── LR tests and bootstrap p-values ──────────────────────────────
        lr_row = {'asset': asset_name}

        _NULL_MODEL_STR = {
            'level':  'level_restricted',
            'memory': 'memory_restricted',
            'joint':  'joint_symmetric',
        }

        for restriction, lr_name, df in RESTRICTIONS:
            null_res = results[_NULL_MODEL_STR[restriction]]
            lr_stat, pval_asym = lr_test(unres.loglik, null_res.loglik, df)

            print(f"\n    {lr_name} = {lr_stat:.4f}  (asym p = {pval_asym:.4f}, df = {df})")
            lr_row[f'lr_{restriction}']        = lr_stat
            lr_row[f'pval_asym_{restriction}'] = pval_asym

            # Gaussian bootstrap — pass pre-fitted null to skip redundant re-fit
            scheme = 'gaussian'
            print(f"      bootstrap [{scheme}] B={B} ...", end=" ", flush=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                boot_g = bootstrap_lr_pvalue(
                    r, restriction=restriction, lr_observed=lr_stat,
                    B=B, scheme=scheme, seed=42,
                    n_starts=N_STARTS_MAIN, n_starts_boot=N_STARTS_BOOT,
                    p=1.2, tau=1.0,
                    null_result=null_res,   # reuse from main estimation
                    verbose=True,
                )
            pv_g = boot_g['p_value']
            cf_g = boot_g['converged_fraction']
            print(f"p_boot = {pv_g:.4f}  (converged {cf_g:.1%})")
            lr_row[f'pval_boot_gaussian_{restriction}'] = pv_g

            # Symmetrized bootstrap (fewer reps — slower residual replay)
            scheme = 'symmetrized'
            print(f"      bootstrap [{scheme}] B={B_SYMM} ...", end=" ", flush=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                boot_s = bootstrap_lr_pvalue(
                    r, restriction=restriction, lr_observed=lr_stat,
                    B=B_SYMM, scheme=scheme, seed=42,
                    n_starts=N_STARTS_MAIN, n_starts_boot=N_STARTS_BOOT,
                    p=1.2, tau=1.0,
                    null_result=null_res,
                    verbose=True,
                )
            pv_s = boot_s['p_value']
            cf_s = boot_s['converged_fraction']
            print(f"p_boot = {pv_s:.4f}  (converged {cf_s:.1%})")
            lr_row[f'pval_boot_symmetrized_{restriction}'] = pv_s

        lr_rows.append(lr_row)
        print()

        # ── Persistence summaries ─────────────────────────────────────────
        prm = unres.params
        pers = compute_persistence(prm, n_lags=251)

        xi_p  = prm.xi_plus
        xi_m  = prm.xi_minus
        rho_p = pers['rho_plus']
        rho_m = pers['rho_minus']
        gp    = prm.gamma_plus
        gm    = prm.gamma_minus

        print(f"    Persistence (unrestricted):")
        print(f"      xi+={xi_p:.6f}  xi-={xi_m:.6f}")
        print(f"      gamma+={gp:.4f}  gamma-={gm:.4f}")
        print(f"      rho+={rho_p:.6f}  rho-={rho_m:.6f}")
        print(f"      half-life+={pers['half_life_plus']:.1f}  "
              f"half-life-={pers['half_life_minus']:.1f}")

        persistence_rows.append({
            'asset':    asset_name,
            'xi_plus':  xi_p,
            'xi_minus': xi_m,
            'rho_plus': rho_p,
            'rho_minus': rho_m,
            'gamma_plus':  gp,
            'gamma_minus': gm,
        })

        # ── Response curves ───────────────────────────────────────────────
        p   = prm.p
        tau = prm.tau
        g_plus  = response_curve(xi_p, gp, p, tau)
        g_minus = response_curve(xi_m, gm, p, tau)
        k_arr   = np.arange(len(g_plus))

        ax = axes_flat[plot_idx]
        ax.plot(k_arr, g_plus,  color='steelblue', linewidth=1.5,
                label=r'$g_+(k)$')
        ax.plot(k_arr, g_minus, color='tomato',    linewidth=1.5,
                linestyle='--', label=r'$g_-(k)$')
        ax.set_title(asset_name, fontsize=11)
        ax.set_xlabel('Lag $k$', fontsize=9)
        ax.set_ylabel(r'$g_s(k)$', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # ── Save CSVs ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("Saving results ...")

    path_params = os.path.join(RESULTS_DIR, 'exp1_parameters.csv')
    pd.DataFrame(params_rows).to_csv(path_params, index=False)
    print(f"  {path_params}")

    path_lr = os.path.join(RESULTS_DIR, 'exp1_lr_tests.csv')
    pd.DataFrame(lr_rows).to_csv(path_lr, index=False)
    print(f"  {path_lr}")

    path_pers = os.path.join(RESULTS_DIR, 'exp1_persistence.csv')
    pd.DataFrame(persistence_rows).to_csv(path_pers, index=False)
    print(f"  {path_pers}")

    # ── Save response-curve figure ────────────────────────────────────────
    fig.suptitle(
        r'ALM-GARCH Response Curves  $g_s(k) = \xi_s \left[\frac{\gamma_s}{\gamma_s + k\tau}\right]^p$',
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path_fig = os.path.join(RESULTS_DIR, 'exp1_response_curves.png')
    fig.savefig(path_fig, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  {path_fig}")

    print("\nDone — Experiment 1 complete.")


if __name__ == '__main__':
    main()
