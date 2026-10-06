"""
make_figures_tables.py -- builds every figure/table of the v2 paper from reports/bias_law/*.csv
(and reports/or_dml_benchmark_summary.csv).  Output: paper/plots/bl_*.pdf, paper/generated/*.tex, reports/bias_law/summary_numbers.json

    python src/bias_law/make_figures_tables.py
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
R = os.path.join(ROOT, 'reports', 'bias_law')
GEN = os.path.join(ROOT, 'paper', 'generated')
PLT = os.path.join(ROOT, 'paper', 'plots')
os.makedirs(GEN, exist_ok=True)
os.makedirs(PLT, exist_ok=True)

C = {'blue': '#0072B2', 'orange': '#E69F00', 'green': '#009E73', 'red': '#D55E00', 'purple': '#CC79A7', 'grey': '#555555', 'sky': '#56B4E9'}
plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': 0.25,
                     'figure.dpi': 200, 'savefig.bbox': 'tight', 'font.family': 'serif'})
NUM = {}


def tex_escape(s):
    return str(s).replace('_', r'\_').replace('%', r'\%')


def fig1():
    s = pd.read_csv(os.path.join(R, 'e1_learned_hmm_summary.csv'))
    s['dz'] = s.dz.round(2)
    b = pd.read_csv(os.path.join(R, 'e1_baselines_summary.csv'))
    b['dz'] = b.dz.round(2)
    x = s[(s['mode'] == 'smooth') & (s.nuis == 'lin')]
    fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.35))
    cols = {0.5: C['red'], 0.8: C['blue'], 1.2: C['green']}
    for dz, g in x.groupby('dz'):
        g = g.sort_values('dT')
        ax[0].errorbar(g.dT, g.bias, yerr=1.96 * g.bias_se, fmt='o', ms=3, color=cols[dz], lw=0.8, label=f'$\\Delta_Z={dz}$ (measured)')
        ax[0].plot(g.dT, g.law_hat, '-', color=cols[dz], lw=1.0, alpha=0.9)
        ax[1].plot(g.var_Ttilde, g.bias, 'o-', ms=3, color=cols[dz], lw=0.9)
    ax[0].set_xscale('log')
    ax[0].set_xlabel('treatment shift $\\Delta_T$ (confounding strength)')
    ax[0].set_ylabel('bias of $\\hat\\theta$')
    ax[0].set_title('(a) bias is hump-shaped; line = law with $\\hat v$', fontsize=7.5)
    ax[0].legend(fontsize=6, frameon=False, loc='upper left')
    ax[1].set_xscale('log')
    ax[1].set_xlabel('$\\mathrm{Var}(\\tilde T)$ ($\\lambda_{\\min}$ analogue)')
    ax[1].set_title('(b) bias vs conditioning: not monotone', fontsize=7.5)
    z = b[b.dz == 0.8]
    sm = x[x.dz == 0.8].sort_values('dT')
    ax[2].plot(z[z.nuis == 'standard_dml'].dT, z[z.nuis == 'standard_dml'].bias, 's-', ms=3, color=C['grey'], label='DML on $X$ only')
    ax[2].plot(z[z.nuis == 'hard_fe'].dT, z[z.nuis == 'hard_fe'].bias, '^-', ms=3, color=C['orange'], label='hard regime FE')
    ax[2].plot(sm.dT, sm.bias, 'o-', ms=3, color=C['blue'], label='soft posterior ($\\hat\\gamma$)')
    ax[2].plot(z[z.nuis == 'oracle_state'].dT, z[z.nuis == 'oracle_state'].bias, 'd-', ms=3, color=C['green'], label='oracle state')
    ax[2].set_xscale('log')
    ax[2].set_xlabel('treatment shift $\\Delta_T$')
    ax[2].set_title('(c) comparators, $\\Delta_Z=0.8$', fontsize=7.5)
    ax[2].legend(fontsize=6, frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(PLT, 'bl_fig1_bias_law.pdf'))
    plt.close()

    # summary numbers
    allc = s.copy()
    for c in ('law_true', 'law_hat', 'law_obs'):
        big = allc[allc.bias.abs() > 0.05]
        NUM[f'e1_corr_{c}'] = float(np.corrcoef(allc.bias, allc[c])[0, 1])
        NUM[f'e1_rmse_{c}'] = float(np.sqrt(np.mean((allc.bias - allc[c]) ** 2)))
        NUM[f'e1_medrel_{c}'] = float(np.median(np.abs(big[c] - big.bias) / big.bias.abs()))
    NUM['e1_cells'] = int(len(allc))
    raw = pd.read_csv(os.path.join(R, 'e1_learned_hmm_raw.csv'))
    xr = raw[(raw['mode'] == 'smooth') & (raw.nuis == 'lin')]
    for f in (0.0, 0.5, 1.0, 2.0):
        NUM[f'adj_mae_bfac_{f}'] = float((xr.bias - f * xr.law_obs).abs().mean())
    # peak location check
    pk = []
    for (dz, nu, md), g in allc.groupby(['dz', 'nuis', 'mode']):
        g = g.sort_values('dT')
        pk.append(dict(dz=dz, nuis=nu, mode=md, v_hat=float(g.v_hat.mean()), pred_peak=float(1 / np.sqrt(g.v_hat.mean())),
                       obs_peak=float(g.dT.values[g.bias.values.argmax()]), pred_max=float(3.0 * np.sqrt(g.v_hat.mean()) / 2), obs_max=float(g.bias.max())))
    pd.DataFrame(pk).to_csv(os.path.join(R, 'e1_peak_check.csv'), index=False)


def table_verification():
    s = pd.read_csv(os.path.join(R, 'e1_learned_hmm_summary.csv'))
    s2 = pd.read_csv(os.path.join(R, 'e2_three_state_summary.csv'))
    rows = []
    for (md, nu), g in s.groupby(['mode', 'nuis']):
        big = g[g.bias.abs() > 0.05]
        rows.append((f'$K{{=}}2$, {md}, {nu}', len(g), np.corrcoef(g.bias, g.law_hat)[0, 1], np.sqrt(np.mean((g.bias - g.law_hat) ** 2)),
                     np.median(np.abs(big.law_hat - big.bias) / big.bias.abs()) if len(big) else np.nan,
                     np.corrcoef(g.bias, g.law_true)[0, 1]))
    big = s2[s2.bias.abs() > 0.05]
    rows.append(('$K{=}3$, smooth, lin', len(s2), np.corrcoef(s2.bias, s2.law_hat)[0, 1], np.sqrt(np.mean((s2.bias - s2.law_hat) ** 2)),
                 np.median(np.abs(big.law_hat - big.bias) / big.bias.abs()) if len(big) else np.nan, np.corrcoef(s2.bias, s2.law_true)[0, 1]))
    NUM['e2_corr_law_hat'] = float(np.corrcoef(s2.bias, s2.law_hat)[0, 1])
    NUM['e2_rmse_law_hat'] = float(np.sqrt(np.mean((s2.bias - s2.law_hat) ** 2)))
    lines = [r'\begin{tabular}{lrrrrr}', r'\toprule', r'Setting & cells & corr($\hat\theta{-}\theta$, law$(\hat v)$) & RMSE & med.\ rel.\ err. & corr (law$(v_N)$) \\', r'\midrule']
    for r in rows:
        lines.append(f'{r[0]} & {r[1]} & {r[2]:.3f} & {r[3]:.3f} & {100 * r[4]:.1f}\\% & {r[5]:.3f} \\\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    open(os.path.join(GEN, 'tab_verification.tex'), 'w').write('\n'.join(lines))


def table_table1dgp():
    s = pd.read_csv(os.path.join(R, 'e3_table1_dgp_summary.csv'))
    lines = [r'\begin{tabular}{rrrrrrr}', r'\toprule', r'$\Delta_Z$ & DML on $X$ & soft $\hat\gamma$ & oracle $S$ & excess bias & law$(v_N)$ & law$(\hat v)$ \\', r'\midrule']
    for _, r in s.iterrows():
        lines.append(f'{r.dz:.1f} & {r.th_std:.2f} & {r.th_soft:.3f} & {r.th_oracle:.3f} & {r.excess:.3f} & {r.law_true:.3f} & {r.law_hat:.3f} \\\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    open(os.path.join(GEN, 'tab_table1dgp.tex'), 'w').write('\n'.join(lines))
    NUM['e3_max_abs_err_law_true'] = float((s.excess - s.law_true).abs().max())


def table_calibration():
    s = pd.read_csv(os.path.join(R, 'e4_calibration_summary.csv'))
    lines = [r'\begin{tabular}{rrrrrrr}', r'\toprule', r'$\tau$ & bias & law$(\hat v)$ & law$(v_N)$ & $\hat v$ & $v_N$ & $\sqrt{CE}{+}CE$ \\', r'\midrule']
    for _, r in s.iterrows():
        lines.append(f'{r.tau:.2f} & {r.bias:.3f} & {r.law_hat:.3f} & {r.law_true:.3f} & {r.v_hat:.3f} & {r.v_N:.3f} & {r.bound:.3f} \\\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    open(os.path.join(GEN, 'tab_calibration.tex'), 'w').write('\n'.join(lines))
    NUM['e4_bound_holds_min'] = float(s.bound_holds_frac.min())


def table_e7():
    s = pd.read_csv(os.path.join(R, 'e7_corrected_conditioning.csv'))
    a = s[s.nuisance.str.startswith('X_only')].sort_values('dT')
    b = s[s.nuisance.str.startswith('X+regime')].sort_values('dT')
    lines = [r'\begin{tabular}{rrrrr}', r'\toprule', r'$\Delta_T$ & \multicolumn{2}{c}{nuisances on $X$ only} & \multicolumn{2}{c}{nuisances on $(X,S)$} \\', r' & $\lambda_{\min}(\hat J)$ & $\|\hat\theta-\theta\|_2$ & $\lambda_{\min}(\hat J)$ & $\|\hat\theta-\theta\|_2$ \\', r'\midrule']
    for (_, r1), (_, r2) in zip(a.iterrows(), b.iterrows()):
        lines.append(f'{r1.dT:.1f} & {r1.lam_min:.2f} & {r1.err_l2:.2f} & {r2.lam_min:.2f} & {r2.err_l2:.2f} \\\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    open(os.path.join(GEN, 'tab_e7.tex'), 'w').write('\n'.join(lines))


def _cities_tables(s, full_name, comp_name):
    full = [r'\begin{tabular}{llrrrrrrr}', r'\toprule', r'City & screen & $N$ & $\hat\theta$ [90\% CI] & $\hat v$ & $\hat a$ & $\mathrm{Var}(\tilde T)$ & $b^*$ (SD of $\tilde Y$) [90\% CI] & $b^*/|\hat c_Y|$ \\', r'\midrule']
    comp = [r'\begin{tabular}{llrrr}', r'\toprule', r'City & screened & $\hat\theta$ [90\% CI] & $b^*$ (SD) [90\% CI] & $b^*/|\hat c_Y|$ \\', r'\midrule']
    for _, r in s.iterrows():
        sc = 'yes' if r.frozen_screen else 'no'
        full.append(f"{r.city} & {sc} & {int(r.N)} & {r.theta:.3f} [{r.theta_lo:.3f}, {r.theta_hi:.3f}] & {r.v_hat:.3f} & {r.a_hat:.3f} & {r.var_Ttilde:.2f} & {r.b_star_sd:.2f} [{r.b_star_sd_lo:.2f}, {r.b_star_sd_hi:.2f}] & {r.b_star_over_obs:.0f} \\\\")
        if not (r.city == 'Delhi' and r.frozen_screen):
            comp.append(f"{r.city} & {sc} & {r.theta:.2f} [{r.theta_lo:.2f}, {r.theta_hi:.2f}] & {r.b_star_sd:.1f} [{r.b_star_sd_lo:.1f}, {r.b_star_sd_hi:.0f}] & {r.b_star_over_obs:.0f} \\\\")
    full += [r'\bottomrule', r'\end{tabular}']
    comp += [r'\bottomrule', r'\end{tabular}']
    open(os.path.join(GEN, full_name), 'w').write('\n'.join(full))
    open(os.path.join(GEN, comp_name), 'w').write('\n'.join(comp))


def table_cities_and_fig2():
    legacy = os.path.join(R, 'e5_real_city_sensitivity.csv')          # 2-hourly series from the legacy pipeline (round-half-even artifact)
    fixed = os.path.join(R, 'x23_real_hourly_fix.csv')                # true hourly grid
    if os.path.exists(legacy):
        _cities_tables(pd.read_csv(legacy), 'tab_cities_2h_full.tex', 'tab_cities_2h.tex')
    p = fixed if os.path.exists(fixed) else legacy
    if not os.path.exists(p):
        return
    s = pd.read_csv(p)
    _cities_tables(s, 'tab_cities_full.tex', 'tab_cities.tex')
    q = s[~s.frozen_screen].reset_index(drop=True)
    fig, ax = plt.subplots(1, 1, figsize=(3.4, 1.9))
    for i, r in q.iterrows():
        informative = abs(r.theta) > 2 * (r.theta_hi - r.theta_lo) / 3.29
        col = C['blue'] if informative else C['grey']
        ax.plot([r.b_star_sd_lo, r.b_star_sd_hi], [i, i], color=col, lw=2, alpha=0.5)
        ax.plot(r.b_star_sd, i, 'o', color=col, ms=5)
    ax.axvline(1, color=C['red'], lw=0.8, ls='--')
    ax.set_xscale('log')
    ax.set_yticks(range(len(q)))
    ax.set_yticklabels(q.city)
    ax.invert_yaxis()
    ax.set_xlabel('robustness value $b^*$ (SD of $\\tilde Y$); dashed: 1 SD')
    plt.tight_layout()
    plt.savefig(os.path.join(PLT, 'bl_fig2_cities.pdf'))
    plt.close()


def table_benchmark_full():
    d = pd.read_csv(os.path.join(ROOT, 'reports', 'or_dml_benchmark_summary.csv'))
    lines = [r'\begin{tabular}{rlrrrr}', r'\toprule', r'$\Delta_Z$ & Estimator & mean $|$bias$|$ & median $|$bias$|$ & RMSE & cover.\ SATE \\', r'\midrule']
    for dz, g in d.groupby('Delta_Z'):
        for i, (_, r) in enumerate(g.iterrows()):
            lines.append(f"{dz if i == 0 else ''} & {tex_escape(r.Method)} & {r.Abs_Bias:.3f} & {r.Median_Abs_Bias:.3f} & {r.RMSE:.3f} & {r.Coverage_SATE_95_Pct:.0f}\\% \\\\")
        lines.append(r'\midrule')
    lines[-1] = r'\bottomrule'
    lines.append(r'\end{tabular}')
    open(os.path.join(GEN, 'tab_benchmark_full.tex'), 'w').write('\n'.join(lines))


def table_hac():
    p = os.path.join(R, 'e6_hac_lag_sensitivity.csv')
    if not os.path.exists(p):
        return
    d = pd.read_csv(p)
    lines = [r'\begin{tabular}{llrrrrr}', r'\toprule', r'City & regime & $\hat\theta$ & SE (12) & SE (24) & SE (72) & SE (168) \\', r'\midrule']
    for (city, reg), g in d.groupby(['city', 'regime'], sort=False):
        g = g.set_index('lag')
        lines.append(f"{city} & {reg} & {g.theta.iloc[0]:.3f} & " + ' & '.join(f'{g.se[l]:.3f}' for l in (12, 24, 72, 168)) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    open(os.path.join(GEN, 'tab_hac.tex'), 'w').write('\n'.join(lines))
    # p-values at lag 12 and 168
    NUM['hac_p'] = {f'{c}_{r}': {int(l): float(g.p[g.lag == l].iloc[0]) for l in (12, 72, 168)} for (c, r), g in d.groupby(['city', 'regime'])}


if __name__ == '__main__':
    fig1()
    table_verification()
    table_table1dgp()
    table_calibration()
    table_e7()
    table_cities_and_fig2()
    table_benchmark_full()
    table_hac()
    json.dump(NUM, open(os.path.join(R, 'summary_numbers.json'), 'w'), indent=1)
    print(json.dumps(NUM, indent=1))
