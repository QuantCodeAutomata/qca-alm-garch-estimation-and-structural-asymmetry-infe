"""
Experiment 3: Robustness Analysis
==================================
Three robustness checks:
  1. Fixed-p sensitivity  — re-estimate with p=2.0 vs baseline p=1.2
  2. Stability-constrained estimation — penalise M*(θ) > -1e-3 during QML
  3. xi_+ floor sensitivity — enforce xi_+ ≥ floor for SP500 and FTSE100

Outputs (all in results/)
-------------------------
  exp3_fixed_p_sensitivity.csv   — params and LR stats for p=1.2 and p=2.0
  exp3_floor_sensitivity.csv     — LR_mem at different xi_+ floors (SP500, FTSE100)
  exp3_constrained_refits.csv    — constrained vs unconstrained log-likelihoods
  exp3_robustness_plot.png       — 2×2 summary figure
"""

import sys
sys.path.insert(0, '/workspace/project')
sys.path.insert(0, '/workspace/project/src')

import os
import warnings
import numpy as np
import pandas as pd
from scipy import optimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.alm_garch import (
    ALMGARCHParams, ALMGARCHResult,
    estimate_alm_garch, lr_test, alm_garch_filter, alm_garch_loglik,
)
from src.stability import compute_stability_certificate
from data.data_loader import load_all_assets

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
os.makedirs(RESULTS_DIR, exist_ok=True)

PARAMS_CSV    = os.path.join(RESULTS_DIR, 'exp1_parameters.csv')
LR_CSV        = os.path.join(RESULTS_DIR, 'exp1_lr_tests.csv')
STAB_CSV      = os.path.join(RESULTS_DIR, 'exp2_stability_table.csv')

# ---------------------------------------------------------------------------
# Log-space bounds for the unrestricted model theta = [log_mu, log_xi+, log_xi-, log_gp, log_gm]
# ---------------------------------------------------------------------------

_UNRES_BOUNDS = [(-20., 2.), (-10., 6.), (-10., 6.), (-2., 8.), (-2., 8.)]


def _unpack_theta(theta):
    """Map log-space theta → (mu, xi_plus, xi_minus, gamma_plus, gamma_minus)."""
    e = np.exp(theta)
    return e[0], e[1], e[2], e[3], e[4]


# ---------------------------------------------------------------------------
# Helper: multi-start unrestricted estimation with custom theta bounds
# ---------------------------------------------------------------------------

def _estimate_unrestricted(
    r: np.ndarray,
    bounds: list,
    p: float = 1.2,
    tau: float = 1.0,
    n_starts: int = 10,
    seed: int = 42,
    nll_fn=None,          # custom neg-loglik; defaults to alm_garch_loglik
) -> ALMGARCHResult:
    """Multi-start unrestricted ALM-GARCH with arbitrary bounds / neg-loglik."""
    if nll_fn is None:
        def nll_fn(theta):
            return alm_garch_loglik(theta, r, model='unrestricted', p=p, tau=tau)

    sample_var = float(np.var(r[np.isfinite(r)]))
    log_sv = np.log(max(sample_var, 1e-8))

    starts = [
        np.array([log_sv - 4.0, np.log(0.05), np.log(0.05),
                  np.log(50.0), np.log(50.0)]),
        np.array([log_sv - 3.5, np.log(0.03), np.log(0.10),
                  np.log(20.0), np.log(5.0)]),
        np.array([log_sv - 4.5, np.log(0.08), np.log(0.08),
                  np.log(10.0), np.log(10.0)]),
    ]

    rng = np.random.RandomState(seed)
    for _ in range(n_starts):
        # Clip random starts to satisfy bounds
        lo = [b[0] for b in bounds]
        hi = [b[1] for b in bounds]
        pt = [rng.uniform(lo[j], hi[j]) for j in range(5)]
        starts.append(np.array(pt))

    # Clip structured starts to bounds
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    starts = [np.clip(s, lo, hi) for s in starts]

    best_nll   = np.inf
    best_theta = None

    for theta0 in starts:
        for method, kwargs in [
            ('L-BFGS-B', dict(bounds=bounds,
                              options={'maxiter': 600, 'ftol': 1e-11, 'gtol': 1e-7})),
            ('Nelder-Mead', dict(options={'maxiter': 2000, 'xatol': 1e-9,
                                          'fatol': 1e-9, 'adaptive': True})),
        ]:
            try:
                res = optimize.minimize(nll_fn, theta0, method=method, **kwargs)
                if np.isfinite(res.fun) and res.fun < best_nll:
                    best_nll   = res.fun
                    best_theta = res.x.copy()
            except Exception:
                pass

    # Polish
    if best_theta is not None:
        try:
            res_pol = optimize.minimize(
                nll_fn, best_theta, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-9},
            )
            if np.isfinite(res_pol.fun) and res_pol.fun < best_nll:
                best_theta = res_pol.x.copy()
        except Exception:
            pass

    if best_theta is None:
        best_theta = starts[0]

    mu, xi_p, xi_m, gp, gm = _unpack_theta(best_theta)
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
# Stability-penalized neg-loglik
# ---------------------------------------------------------------------------

def _make_stability_penalized_nll(r, p, tau, M_threshold=-1e-3,
                                   penalty_weight=1e6, n_c_grid=100, n_quad=32):
    """Return a neg-loglik function that adds a large penalty if M*(θ) > threshold."""
    def penalized_nll(theta):
        nll = alm_garch_loglik(theta, r, model='unrestricted', p=p, tau=tau)
        if nll > 1e9:
            return nll
        _, xi_p, xi_m, gp, gm = _unpack_theta(theta)
        try:
            cert = compute_stability_certificate(
                xi_plus=float(xi_p), xi_minus=float(xi_m),
                gamma_plus=float(gp), gamma_minus=float(gm),
                p=p, tau=tau,
                n_c_grid=n_c_grid, n_quad=n_quad,
            )
            M_star = cert['M_star']
        except Exception:
            M_star = 0.0   # assume worst-case if computation fails
        if M_star > M_threshold:
            nll += penalty_weight * (M_star - M_threshold) ** 2
        return nll

    return penalized_nll


# ---------------------------------------------------------------------------
# Load or re-estimate data helpers
# ---------------------------------------------------------------------------

def load_returns_all():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return load_all_assets()


def _load_exp1_lr():
    if os.path.exists(LR_CSV):
        df = pd.read_csv(LR_CSV).set_index('asset')
        return df
    return None


def _load_exp2_stability():
    if os.path.exists(STAB_CSV):
        df = pd.read_csv(STAB_CSV, index_col=0)
        return df
    return None


def _load_exp1_params():
    """Return full exp1 parameters DataFrame (all models × assets), or None."""
    if not os.path.exists(PARAMS_CSV):
        return None
    return pd.read_csv(PARAMS_CSV)


# ---------------------------------------------------------------------------
# Section 1: Fixed-p sensitivity  (p=1.2 vs p=2.0)
# ---------------------------------------------------------------------------

_NULL_MODEL = {
    'level':  'level_restricted',
    'memory': 'memory_restricted',
    'joint':  'joint_symmetric',
}
_DF = {'level': 1, 'memory': 1, 'joint': 2}


def _lookup_loglik(df_all: pd.DataFrame, asset: str, model: str) -> float | None:
    """Return stored loglik for (asset, model), or None if not found."""
    if df_all is None:
        return None
    mask = (df_all['asset'] == asset) & (df_all['model'] == model)
    rows = df_all[mask]
    if rows.empty:
        return None
    return float(rows.iloc[0]['loglik'])


def _lookup_params(df_all: pd.DataFrame, asset: str, model: str):
    """Return ALMGARCHParams from stored row, or None."""
    if df_all is None:
        return None
    mask = (df_all['asset'] == asset) & (df_all['model'] == model)
    rows = df_all[mask]
    if rows.empty:
        return None
    row = rows.iloc[0]
    return ALMGARCHParams(
        mu=float(row['mu']),
        xi_plus=float(row['xi_plus']),
        xi_minus=float(row['xi_minus']),
        gamma_plus=float(row['gamma_plus']),
        gamma_minus=float(row['gamma_minus']),
        p=float(row['p']),
        tau=float(row['tau']),
    )


def section_fixed_p(assets: dict, df_params_12=None) -> pd.DataFrame:
    """Re-estimate with p=2.0; compare LR stats to p=1.2 baseline.

    For p=1.2, logliks are loaded from exp1_parameters.csv where available
    (all four models), avoiding redundant optimisations.
    """
    rows = []

    for asset_name, ret in assets.items():
        r = ret.values.astype(float)
        print(f"  {asset_name} ...", end=" ", flush=True)

        for p_val in (1.2, 2.0):
            # ── p=1.2: prefer cached exp1 results ──
            if p_val == 1.2 and df_params_12 is not None:
                prm          = _lookup_params(df_params_12, asset_name, 'unrestricted')
                loglik_unres = _lookup_loglik(df_params_12, asset_name, 'unrestricted')

                restricted_logliks = {}
                for restr, model_str in _NULL_MODEL.items():
                    ll = _lookup_loglik(df_params_12, asset_name, model_str)
                    if ll is not None:
                        restricted_logliks[restr] = ll
                    else:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            res_r = estimate_alm_garch(
                                r, model=model_str, p=1.2, tau=1.0,
                                n_starts=4, seed=42,
                            )
                        restricted_logliks[restr] = res_r.loglik

                # Fall back to estimation if params not available in cache
                if prm is None or loglik_unres is None:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        res_u = estimate_alm_garch(
                            r, model='unrestricted', p=1.2, tau=1.0,
                            n_starts=5, seed=42,
                        )
                    prm          = res_u.params
                    loglik_unres = res_u.loglik

            else:
                # ── p=2.0 (or no cache): estimate all four models fresh ──
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res_u = estimate_alm_garch(
                        r, model='unrestricted', p=p_val, tau=1.0,
                        n_starts=5, seed=42,
                    )
                prm          = res_u.params
                loglik_unres = res_u.loglik

                restricted_logliks = {}
                for restr, model_str in _NULL_MODEL.items():
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        res_r = estimate_alm_garch(
                            r, model=model_str, p=p_val, tau=1.0,
                            n_starts=4, seed=42,
                        )
                    restricted_logliks[restr] = res_r.loglik

            row = {
                'asset':       asset_name,
                'p':           p_val,
                'mu':          prm.mu,
                'xi_plus':     prm.xi_plus,
                'xi_minus':    prm.xi_minus,
                'gamma_plus':  prm.gamma_plus,
                'gamma_minus': prm.gamma_minus,
                'loglik':      loglik_unres,
            }
            for restr in ('level', 'memory', 'joint'):
                lr_s, pv = lr_test(loglik_unres, restricted_logliks[restr], _DF[restr])
                row[f'lr_{restr}']       = lr_s
                row[f'pval_{restr}']     = pv
                row[f'reject05_{restr}'] = (pv < 0.05)

            rows.append(row)

        print("done")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Section 2: Stability-constrained re-estimation
# ---------------------------------------------------------------------------

def section_constrained(assets: dict, df_params_12=None, df_stab=None) -> pd.DataFrame:
    """Re-estimate each asset with stability penalty, compare to unconstrained.

    Cached exp1 logliks and exp2 M* values are used where available so that
    the unconstrained model does not need to be re-fitted.
    """
    rows = []

    for asset_name, ret in assets.items():
        r = ret.values.astype(float)
        print(f"  {asset_name} ...", end=" ", flush=True)

        # ── Unconstrained loglik: prefer cached exp1 value ───────────────
        loglik_u = _lookup_loglik(df_params_12, asset_name, 'unrestricted')
        prm_u    = _lookup_params(df_params_12, asset_name, 'unrestricted')

        if loglik_u is None or prm_u is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res_u = estimate_alm_garch(
                    r, model='unrestricted', p=1.2, tau=1.0, n_starts=5, seed=42
                )
            loglik_u = res_u.loglik
            prm_u    = res_u.params

        # ── M* for unconstrained: prefer cached exp2 value ───────────────
        if df_stab is not None and asset_name in df_stab.index:
            M_star_u = float(df_stab.loc[asset_name, 'M_star'])
        else:
            cert_u = compute_stability_certificate(
                xi_plus=float(prm_u.xi_plus),
                xi_minus=float(prm_u.xi_minus),
                gamma_plus=float(prm_u.gamma_plus),
                gamma_minus=float(prm_u.gamma_minus),
                p=1.2, tau=1.0,
            )
            M_star_u = cert_u['M_star']

        # Stability-constrained estimation
        nll_pen = _make_stability_penalized_nll(
            r, p=1.2, tau=1.0, M_threshold=-1e-3,
            penalty_weight=1e6, n_c_grid=100, n_quad=32
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res_c = _estimate_unrestricted(
                r, bounds=_UNRES_BOUNDS, p=1.2, tau=1.0,
                n_starts=5, seed=42, nll_fn=nll_pen
            )

        # M* for constrained result (full grid for accuracy)
        cert_c = compute_stability_certificate(
            xi_plus=float(res_c.params.xi_plus),
            xi_minus=float(res_c.params.xi_minus),
            gamma_plus=float(res_c.params.gamma_plus),
            gamma_minus=float(res_c.params.gamma_minus),
            p=1.2, tau=1.0,
        )
        M_star_c = cert_c['M_star']

        constraint_binding = (M_star_u > -1e-3)
        loglik_c = res_c.loglik

        print(
            f"loglik_u={loglik_u:.4f}  loglik_c={loglik_c:.4f}  "
            f"M*_u={M_star_u:+.5f}  M*_c={M_star_c:+.5f}  "
            f"binding={constraint_binding}"
        )

        rows.append({
            'asset':               asset_name,
            'loglik_unconstrained': loglik_u,
            'loglik_constrained':   loglik_c,
            'loglik_diff':          loglik_u - loglik_c,
            'M_star_unconstrained': M_star_u,
            'M_star_constrained':   M_star_c,
            'constraint_binding':   constraint_binding,
            'xi_plus_u':            prm_u.xi_plus,
            'xi_minus_u':           prm_u.xi_minus,
            'gamma_plus_u':         prm_u.gamma_plus,
            'gamma_minus_u':        prm_u.gamma_minus,
            'xi_plus_c':            res_c.params.xi_plus,
            'xi_minus_c':           res_c.params.xi_minus,
            'gamma_plus_c':         res_c.params.gamma_plus,
            'gamma_minus_c':        res_c.params.gamma_minus,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Section 3: xi_+ floor sensitivity (SP500 and FTSE100)
# ---------------------------------------------------------------------------

_FLOOR_ASSETS = ['SP500', 'FTSE100']
_FLOORS       = [0.0, 0.01, 0.02, 0.05]


def section_floor(assets: dict) -> pd.DataFrame:
    """Re-estimate with xi_+ >= floor; report LR_mem."""
    rows = []

    for asset_name in _FLOOR_ASSETS:
        if asset_name not in assets:
            print(f"  {asset_name} not found — skipping")
            continue
        r = assets[asset_name].values.astype(float)

        for floor in _FLOORS:
            print(f"  {asset_name}  xi_floor = {floor:.3f} ...", end=" ", flush=True)

            if floor <= 0.0:
                # No floor — regular unrestricted estimation
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res_u = estimate_alm_garch(
                        r, model='unrestricted', p=1.2, tau=1.0,
                        n_starts=5, seed=42
                    )
                loglik_u = res_u.loglik
                prm      = res_u.params
            else:
                # Floor on log(xi_+): lower bound = max(-10, log(floor))
                bounds = list(_UNRES_BOUNDS)
                bounds[1] = (max(-10.0, float(np.log(floor))), 6.0)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res_u = _estimate_unrestricted(
                        r, bounds=bounds, p=1.2, tau=1.0,
                        n_starts=5, seed=42
                    )
                loglik_u = res_u.loglik
                prm      = res_u.params

            # Memory-restricted model (gamma+ = gamma-)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res_mem = estimate_alm_garch(
                    r, model='memory_restricted', p=1.2, tau=1.0,
                    n_starts=4, seed=42
                )

            lr_mem, pval_mem = lr_test(loglik_u, res_mem.loglik, df=1)

            print(
                f"xi+={prm.xi_plus:.5f}  "
                f"LR_mem={lr_mem:.4f}  p={pval_mem:.4f}"
            )

            rows.append({
                'asset':        asset_name,
                'xi_plus_floor': floor,
                'xi_plus_fit':   prm.xi_plus,
                'xi_minus_fit':  prm.xi_minus,
                'gamma_plus':    prm.gamma_plus,
                'gamma_minus':   prm.gamma_minus,
                'loglik_unres':  loglik_u,
                'loglik_mem':    res_mem.loglik,
                'lr_mem':        lr_mem,
                'pval_mem':      pval_mem,
                'reject05_mem':  (pval_mem < 0.05),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Robustness plot
# ---------------------------------------------------------------------------

def make_robustness_plot(df_p: pd.DataFrame, df_floor: pd.DataFrame,
                          df_constr: pd.DataFrame):
    """2×2 summary figure."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ── Panel (0,0): LR statistics under p=1.2 vs p=2.0 ─────────────────
    ax = axes[0, 0]
    assets_list  = df_p['asset'].unique().tolist()
    tests        = ['level', 'memory', 'joint']
    test_labels  = [r'$LR_{lev}$', r'$LR_{mem}$', r'$LR_{sym}$']
    n_tests      = len(tests)
    n_assets_p   = len(assets_list)
    x            = np.arange(n_assets_p)
    bar_width    = 0.35

    for ti, (test, label) in enumerate(zip(tests, test_labels)):
        col = f'lr_{test}'
        vals_12 = [
            float(df_p[(df_p['asset'] == a) & (df_p['p'] == 1.2)][col].values[0])
            for a in assets_list
        ]
        vals_20 = [
            float(df_p[(df_p['asset'] == a) & (df_p['p'] == 2.0)][col].values[0])
            for a in assets_list
        ]
        offset = (ti - 1) * bar_width
        bars12 = ax.bar(x + offset - bar_width / 2, vals_12,
                        width=bar_width / n_tests, label=f'{label} p=1.2',
                        alpha=0.7)
        bars20 = ax.bar(x + offset + bar_width / 2, vals_20,
                        width=bar_width / n_tests, label=f'{label} p=2.0',
                        alpha=0.7, hatch='//')

    ax.set_xticks(x)
    ax.set_xticklabels([a[:6] for a in assets_list], rotation=25, ha='right')
    ax.set_ylabel('LR statistic')
    ax.set_title('LR Statistics: $p=1.2$ vs $p=2.0$')
    ax.legend(fontsize=6, ncol=2, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    # ── Panel (0,1): Floor sensitivity (LR_mem vs xi_+ floor) ────────────
    ax = axes[0, 1]
    colors = {'SP500': 'steelblue', 'FTSE100': 'tomato'}
    for asset_name in _FLOOR_ASSETS:
        sub = df_floor[df_floor['asset'] == asset_name].sort_values('xi_plus_floor')
        if sub.empty:
            continue
        ax.plot(sub['xi_plus_floor'], sub['lr_mem'],
                marker='o', label=asset_name, color=colors.get(asset_name))

    ax.axhline(3.841, color='gray', linestyle='--', linewidth=1,
               label=r'$\chi^2_{0.05}(1)=3.84$')
    ax.set_xlabel(r'$\xi_+$ floor')
    ax.set_ylabel(r'$LR_{mem}$')
    ax.set_title(r'Floor Sensitivity: $LR_{mem}$ vs $\xi_+$ floor')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Panel (1,0): Log-lik loss from stability constraint ───────────────
    ax = axes[1, 0]
    asset_c  = df_constr['asset'].tolist()
    ll_diff  = df_constr['loglik_diff'].tolist()
    colors_c = ['crimson' if b else 'steelblue'
                for b in df_constr['constraint_binding'].tolist()]
    xc = np.arange(len(asset_c))
    ax.bar(xc, ll_diff, color=colors_c, alpha=0.8)
    ax.set_xticks(xc)
    ax.set_xticklabels([a[:6] for a in asset_c], rotation=25, ha='right')
    ax.set_ylabel(r'$\ell_{\mathrm{uncons}} - \ell_{\mathrm{constr}}$')
    ax.set_title('Log-likelihood Loss from Stability Constraint')
    ax.set_ylim(bottom=0)
    # Legend patches
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor='crimson',   alpha=0.8, label='Binding'),
        Patch(facecolor='steelblue', alpha=0.8, label='Non-binding'),
    ], fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # ── Panel (1,1): M* before and after constraint ───────────────────────
    ax = axes[1, 1]
    M_u = df_constr['M_star_unconstrained'].tolist()
    M_c = df_constr['M_star_constrained'].tolist()
    xc  = np.arange(len(asset_c))
    ax.plot(xc, M_u, 'o-', color='tomato',    linewidth=1.5, label='Unconstrained')
    ax.plot(xc, M_c, 's--', color='steelblue', linewidth=1.5, label='Constrained')
    ax.axhline(-1e-3, color='gray',  linestyle=':', linewidth=1, label='Threshold −0.001')
    ax.axhline(0.0,   color='black', linestyle=':', linewidth=0.8)
    ax.set_xticks(xc)
    ax.set_xticklabels([a[:6] for a in asset_c], rotation=25, ha='right')
    ax.set_ylabel('$M^*$')
    ax.set_title('$M^*$ Before and After Stability Constraint')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle('ALM-GARCH Robustness Analysis', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Experiment 3: Robustness Analysis")
    print("=" * 70)

    print("\nLoading asset return data ...")
    assets = load_returns_all()

    # Load baseline params from exp1 (to reuse p=1.2 unrestricted fit)
    df_params_12 = _load_exp1_params()
    df_stab      = _load_exp2_stability()

    if df_params_12 is not None:
        print("  Loaded exp1 parameters for p=1.2 baseline.")
    else:
        print("  exp1_parameters.csv not found; will re-estimate p=1.2 models.")

    if df_stab is not None:
        print("  Loaded exp2 stability table for cached M* values.")
    else:
        print("  exp2_stability_table.csv not found; will re-compute M*.")

    # ── Section 1: Fixed-p sensitivity ───────────────────────────────────
    print("\n" + "─" * 70)
    print("Section 1: Fixed-p sensitivity (p=1.2 vs p=2.0)")
    print("─" * 70)
    df_p = section_fixed_p(assets, df_params_12=df_params_12)

    path_p = os.path.join(RESULTS_DIR, 'exp3_fixed_p_sensitivity.csv')
    df_p.to_csv(path_p, index=False)
    print(f"\n  Saved: {path_p}")

    print("\n  LR statistics p=1.2 vs p=2.0:")
    print(f"  {'Asset':<12} {'Test':<8} {'LR(1.2)':>9} {'LR(2.0)':>9} "
          f"{'p(1.2)':>8} {'p(2.0)':>8}")
    print(f"  {'-'*12} {'-'*8} {'-'*9} {'-'*9} {'-'*8} {'-'*8}")
    for asset_name in assets:
        for test in ('level', 'memory', 'joint'):
            row12 = df_p[(df_p['asset'] == asset_name) & (df_p['p'] == 1.2)].iloc[0]
            row20 = df_p[(df_p['asset'] == asset_name) & (df_p['p'] == 2.0)].iloc[0]
            print(
                f"  {asset_name:<12} {test:<8} "
                f"{row12[f'lr_{test}']:>9.4f} {row20[f'lr_{test}']:>9.4f} "
                f"{row12[f'pval_{test}']:>8.4f} {row20[f'pval_{test}']:>8.4f}"
            )

    # ── Section 2: Stability-constrained estimation ───────────────────────
    print("\n" + "─" * 70)
    print("Section 2: Stability-constrained estimation")
    print("─" * 70)
    df_c = section_constrained(assets, df_params_12=df_params_12, df_stab=df_stab)

    path_c = os.path.join(RESULTS_DIR, 'exp3_constrained_refits.csv')
    df_c.to_csv(path_c, index=False)
    print(f"\n  Saved: {path_c}")

    # ── Section 3: Floor sensitivity ──────────────────────────────────────
    print("\n" + "─" * 70)
    print("Section 3: xi_+ floor sensitivity (SP500, FTSE100)")
    print("─" * 70)
    df_floor = section_floor(assets)

    path_f = os.path.join(RESULTS_DIR, 'exp3_floor_sensitivity.csv')
    df_floor.to_csv(path_f, index=False)
    print(f"\n  Saved: {path_f}")

    print("\n  Floor sensitivity summary (LR_mem):")
    print(f"  {'Asset':<12} {'Floor':>8} {'xi+_fit':>10} {'LR_mem':>9} {'p':>8}")
    print(f"  {'-'*12} {'-'*8} {'-'*10} {'-'*9} {'-'*8}")
    for _, row in df_floor.iterrows():
        print(
            f"  {row['asset']:<12} {row['xi_plus_floor']:>8.3f} "
            f"{row['xi_plus_fit']:>10.6f} "
            f"{row['lr_mem']:>9.4f} {row['pval_mem']:>8.4f}"
        )

    # ── Robustness plot ───────────────────────────────────────────────────
    print("\nGenerating robustness plot ...")
    fig = make_robustness_plot(df_p, df_floor, df_c)
    path_fig = os.path.join(RESULTS_DIR, 'exp3_robustness_plot.png')
    fig.savefig(path_fig, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path_fig}")

    print("\nDone — Experiment 3 complete.")


if __name__ == '__main__':
    main()
