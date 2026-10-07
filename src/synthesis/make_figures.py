"""
make_figures.py -- figures for the synthesized manuscript (all inputs are committed CSVs).

  paper/plots/fig3_estimators.pdf      posterior-adjusted OR-DML vs baselines (frontier and semi-synthetic)
  paper/plots/fig1_phantom_resolution.pdf
      (a) factorial causal error vs proxy error eps for four innovation scales (dots: median of 100 runs;
          lines: exact population law, src/synthesis/closed_form_factorial.py)
      (b) lambda_min of the generated-state Gram matrix vs eps (same cells, same law)
      (c) learned-HMM posteriors: lambda_min of the generated-state Gram vs the oracle Gram (720 runs)

    python src/synthesis/make_figures.py
"""
import os
import sys

import matplotlib
import matplotlib.ticker
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
from src.synthesis.closed_form_factorial import population_error   # noqa: E402

PLOTS = os.path.join(ROOT, 'paper', 'plots')
RAMP = ['#86b6ef', '#3987e5', '#1c5cab', '#0d366b']      # ordinal one-hue ramp (validated, light surface)
INK, MUTED, GRID = '#1f1f1e', '#6b6a64', '#e4e3dd'
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'mathtext.fontset': 'dejavuserif',
                     'font.size': 7.5, 'axes.labelsize': 7.5, 'axes.titlesize': 8, 'legend.fontsize': 6.5,
                     'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'axes.edgecolor': MUTED,
                     'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.linewidth': 0.6})


def style(ax):
    ax.grid(True, which='major', color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


def main():
    cells = pd.read_csv(os.path.join(ROOT, 'reports', 'synthesis', 'closed_form_factorial_cells.csv'))
    lp = pd.read_csv(os.path.join(ROOT, 'reports', 'synthesis', 'learned_posterior_coupled_raw.csv'))
    sig_show = [0.08, 0.15, 0.30, 1.50]
    grid = np.geomspace(0.006, 0.45, 160)
    fig, axes = plt.subplots(1, 3, figsize=(6.75, 1.95), constrained_layout=True)

    ax = axes[0]
    for c, s in zip(RAMP, sig_show):
        sub = cells[(cells.sigma1 == s) & (cells.eps > 0)]
        ax.plot(grid, [population_error(s, e)[0] for e in grid], color=c, lw=1.6)
        ax.plot(sub.eps, sub.err_median, 'o', ms=3.6, color=c, mec='white', mew=0.6)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'proxy error $\varepsilon_\gamma$')
    ax.set_ylabel(r'causal error $\|\hat\theta-\theta^*\|_2$')
    ax.set_title('(a) error is non-monotone in $\\varepsilon_\\gamma$', loc='left', color=INK)
    style(ax)

    ax = axes[1]
    for c, s in zip(RAMP, sig_show):
        sub = cells[(cells.sigma1 == s) & (cells.eps > 0)]
        ax.plot(grid, [population_error(s, e)[1] for e in grid], color=c, lw=1.6, label=f'$\\sigma_1={s:g}$')
        ax.plot(sub.eps, sub.lam_mean, 'o', ms=3.6, color=c, mec='white', mew=0.6)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'proxy error $\varepsilon_\gamma$')
    ax.set_ylabel(r'$\lambda_{\min}(\hat J)$, generated state')
    ax.set_title('(b) apparent information inflates', loc='left', color=INK)
    ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(0.0, 0.87), ncol=2, columnspacing=0.8, handlelength=1.0, borderaxespad=0.2, labelspacing=0.25)
    style(ax)

    ax = axes[2]
    x, y = lp.lam_oracle.values, lp.lam_proxy.values
    ax.scatter(x, y, s=5, color=RAMP[2], alpha=0.45, lw=0)
    lo, hi = min(x.min(), y.min()) * 0.8, max(x.max(), y.max()) * 1.2
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.9, ls='--')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    frac = float((y > x).mean())
    ax.text(0.04, 0.95, f'{100 * frac:.0f}% of runs above\nthe identity', transform=ax.transAxes, va='top',
            color=INK, fontsize=6.5)
    ax.set_xlabel(r'$\lambda_{\min}(J^\star)$, oracle state')
    ax.set_ylabel(r'$\lambda_{\min}(\hat J)$, learned HMM')
    ax.set_title('(c) learned posteriors', loc='left', color=INK)
    style(ax)
    fig.savefig(os.path.join(PLOTS, 'fig1_phantom_resolution.pdf'))
    fig.savefig(os.path.join(PLOTS, 'fig1_phantom_resolution.png'), dpi=220)
    print('wrote fig1_phantom_resolution')
    fig2()
    fig3()


def fig2():
    """Residual-state bias law with learned HMM posteriors (reports/bias_law/e1_*)."""
    s = pd.read_csv(os.path.join(ROOT, 'reports', 'bias_law', 'e1_learned_hmm_summary.csv'))
    b = pd.read_csv(os.path.join(ROOT, 'reports', 'bias_law', 'e1_baselines_summary.csv'))
    s['dz'] = s.dz.round(2); b['dz'] = b.dz.round(2)
    x = s[(s['mode'] == 'smooth') & (s.nuis == 'lin')]
    cols = {0.5: RAMP[0], 0.8: RAMP[1], 1.2: RAMP[3]}
    fig, axes = plt.subplots(1, 3, figsize=(6.75, 1.95), constrained_layout=True)
    for dz, g in x.groupby('dz'):
        g = g.sort_values('dT')
        axes[0].plot(g.dT, g.law_hat, '-', color=cols[dz], lw=1.6)
        axes[0].errorbar(g.dT, g.bias, yerr=1.96 * g.bias_se, fmt='o', ms=3.6, color=cols[dz], mec='white', mew=0.6,
                         lw=0.8, label=f'$\\Delta_Z={dz:g}$')
        axes[1].plot(g.var_Ttilde, g.bias, 'o-', ms=3.6, color=cols[dz], mec='white', mew=0.6, lw=1.4)
    axes[0].set_xscale('log'); axes[0].set_xlabel(r'treatment regime shift $\Delta_T$'); axes[0].set_ylabel(r'bias of $\hat\theta$')
    axes[0].set_title('(a) hump; lines = law with $\\hat v$', loc='left', color=INK)
    axes[0].legend(frameon=False, loc='upper left', handlelength=1.0, labelspacing=0.25)
    axes[1].set_xscale('log'); axes[1].set_xlabel(r'$\mathrm{Var}(\tilde T)$, the $\lambda_{\min}$ analogue')
    axes[1].set_title('(b) conditioning is not monotone', loc='left', color=INK)
    z = b[b.dz == 0.8]
    sm = x[x.dz == 0.8].sort_values('dT')
    series = [('DML on $X$ only', z[z.nuis == 'standard_dml'], '#eda100', 's'),
              ('hard regime label', z[z.nuis == 'hard_fe'], '#eb6834', '^'),
              ('soft posterior feature', sm, '#2a78d6', 'o'),
              ('oracle state', z[z.nuis == 'oracle_state'], '#1baf7a', 'D')]
    for lab, g, c, mk in series:
        g = g.sort_values('dT')
        axes[2].plot(g.dT, g.bias, '-', marker=mk, ms=3.4, color=c, mec='white', mew=0.5, lw=1.4, label=lab)
    axes[2].set_xscale('log'); axes[2].set_xlabel(r'treatment regime shift $\Delta_T$')
    axes[2].set_title('(c) estimators, $\\Delta_Z=0.8$', loc='left', color=INK)
    axes[2].legend(frameon=False, loc='upper left', handlelength=1.2, labelspacing=0.25)
    for ax in axes:
        style(ax)
    fig.savefig(os.path.join(PLOTS, 'fig2_bias_law.pdf'))
    fig.savefig(os.path.join(PLOTS, 'fig2_bias_law.png'), dpi=220)
    print('wrote fig2_bias_law')


def fig3():
    """Posterior-adjusted OR-DML against baselines: (a) ATE bias and (b) regime-effect error along the
    observability frontier (learned HMM posteriors, persistence 0.88), (c) regime-effect error in the
    real-data-calibrated semi-synthetic benchmark."""
    fd = pd.read_csv(os.path.join(ROOT, 'reports', 'synthesis', 'frontier_designs_summary.csv'))
    ss = pd.read_csv(os.path.join(ROOT, 'reports', 'synthesis', 'semisynthetic_hourly_summary.csv'))
    fd = fd[fd.p00 == 0.88]
    # categorical slots in a fixed, validated order (scripts/validate_palette.js: all checks pass, light surface);
    # markers are the secondary encoding
    S = [('Standard DML', 'Standard DML', None, '#eda100', 's', '-'),
         ('Posterior-feature DML', 'Posterior-feature DML', None, '#2a78d6', 'o', '-'),
         ('Hard regime FE', 'Hard regime FE', 'Hard regime FE DML', '#eb6834', '^', '-'),
         ('OR-DML (posterior-adjusted)', 'OR-DML', 'OR-DML (posterior-adjusted)', '#4a3aa7', 'D', '-'),
         ('OR-DML, weighted nuisances', 'OR-DML, weighted nuisances', 'Spectral OR-DML (soft, coupled)', '#e87ba4', 'v', '-'),
         ('Oracle state', 'Oracle-state', None, '#1baf7a', 'P', '--')]
    fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.3), constrained_layout=True)
    for lab, fm, _, c, mk, ls in S:
        g = fd[fd.method == fm].sort_values('dz')
        axes[0].plot(g.dz, g.mean_abs_bias, ls=ls, marker=mk, ms=3.4, color=c, mec='white', mew=0.5, lw=1.4, label=lab)
        if fm in ('Hard regime FE', 'OR-DML', 'OR-DML, weighted nuisances', 'Oracle-state'):
            axes[1].plot(g.dz, g.theta_err_median, ls=ls, marker=mk, ms=3.4, color=c, mec='white', mew=0.5, lw=1.4)
    for ax in axes[:2]:
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xticks([0.2, 0.5, 1, 2, 4]); ax.set_xticklabels(['0.2', '0.5', '1', '2', '4'])
        ax.set_xlabel(r'proxy separation $\Delta_Z$')
    axes[0].set_ylabel('mean |ATE bias|')
    axes[0].set_title('(a) ATE, learned posteriors', loc='left', color=INK)
    axes[1].set_ylabel(r'median $\|\hat\theta-\theta^*\|_2$')
    axes[1].set_title('(b) regime effects', loc='left', color=INK)
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, loc="outside lower center", ncol=3,
               handlelength=1.8, columnspacing=1.2, fontsize=6.2)
    cities = ['Delhi', 'Mumbai', 'Bengaluru', 'Kolkata']
    bars = [s for s in S if s[2] is not None]
    w = 0.26
    for i, (lab, _, sm, c, mk, ls) in enumerate(bars):
        vals = [ss[(ss.city == ct) & (ss.method == sm)].theta_err.iloc[0] for ct in cities]
        axes[2].bar(np.arange(4) + (i - 1) * w, vals, width=w - 0.03, color=c, edgecolor='white', linewidth=0.6,
                    label=lab)
    axes[2].set_xticks(np.arange(4)); axes[2].set_xticklabels([c[:3] for c in cities])
    axes[2].set_ylabel(r'median $\|\hat\theta-\theta^*\|_2$')
    axes[2].set_title('(c) real-data-calibrated, hourly', loc='left', color=INK)
    for ax in axes:
        style(ax)
    axes[2].grid(False, axis='x')
    fig.savefig(os.path.join(PLOTS, 'fig3_estimators.pdf'))
    fig.savefig(os.path.join(PLOTS, 'fig3_estimators.png'), dpi=220)
    print('wrote fig3_estimators')


if __name__ == '__main__':
    main()
