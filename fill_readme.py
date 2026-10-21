"""Fill the <!-- AUTO:name --> blocks of README.md from the notebook's outputs/ folder.

Usage (from the repo root, after the notebook has finished):
    python fill_readme.py                 # reads ./outputs, rewrites ./README.md in place
    python fill_readme.py --out OUT_DIR --readme README.md --n-seeds 5

Every number in the AUTO blocks comes from the CSV/JSON files the notebook writes, so the README
always matches the notebook. Re-running is safe: only the AUTO blocks are replaced.
"""
import argparse, json, re
from pathlib import Path
import pandas as pd

NAMES = {'vgg16': 'VGG16', 'resnet50': 'ResNet50', 'mobilenet_v2': 'MobileNetV2'}
nm = lambda a: NAMES.get(a, a)


def mean_of(s):                                   # "38.42 ± 11.98" -> 38.42
    return float(str(s).split('±')[0])


def md_table(df, index=True):
    df = df.reset_index() if index else df
    head = '| ' + ' | '.join(map(str, df.columns)) + ' |'
    sep = '|' + '|'.join(':-:' if i else '---' for i in range(len(df.columns))) + '|'
    rows = ['| ' + ' | '.join(str(v) for v in r) + ' |' for r in df.itertuples(index=False)]
    return '\n'.join([head, sep, *rows])


def build(out, n_seeds):
    B, F = {}, lambda n: out / n
    summ = pd.read_csv(F('summary_metrics.csv'), index_col=0)
    rank = sorted(summ.index, key=lambda a: -mean_of(summ.loc[a, 'Macro F1 (%)']))
    best, worst = rank[0], rank[-1]
    pv = pd.read_csv(F('pvalues.csv')) if F('pvalues.csv').exists() else None
    strat = pd.read_csv(F('strategy_comparison.csv')) if F('strategy_comparison.csv').exists() else None

    def pair_p(a, b, metric='Macro F1'):
        if pv is None: return None
        r = pv[(pv.metric == metric) & (((pv.A == a) & (pv.B == b)) | ((pv.A == b) & (pv.B == a)))]
        return None if r.empty or pd.isna(r['p-value'].iloc[0]) else float(r['p-value'].iloc[0])

    alpha = 0.05 / 6
    def sig_text(p):
        if p is None: return 'p-value undefined'
        return (f'p = {p:.4f}, significant after Bonferroni' if p < alpha else
                f'p = {p:.4f}, significant only without correction' if p < 0.05 else f'p = {p:.4f}, not significant')

    # ---------------- results summary ----------------
    cols = ['strategy', 'Accuracy (%)', 'Balanced Acc (%)', 'Macro F1 (%)', 'Weighted F1 (%)', 'Top-3 Acc (%)',
            'Params (M)', 'Trainable (M)', 'Latency (ms/img)']
    t = summ[[c for c in cols if c in summ.columns]].loc[rank].copy()
    t.index = [('🏆 ' if a == best else '') + f'**{nm(a)}**' for a in t.index]
    t.index.name = 'Model'
    B['summary'] = (md_table(t) + f'\n\n> Ranked by mean test Macro-F1 (primary metric). SD = sample SD (ddof = 1). '
                    'Latency = forward-pass time per image at batch size 32 (throughput, not single-image latency), measured on the training machine (Apple Silicon, MPS).')

    # ---------------- highlight + conclusion ----------------
    p_bw = pair_p(best, rank[1])
    hl = [f"1. **After W&B Sweep tuning + {n_seeds} seeds per model**, the best model is **🏆{nm(best)}** with test Macro-F1 "
          f"**{summ.loc[best, 'Macro F1 (%)']}%** and accuracy **{summ.loc[best, 'Accuracy (%)']}%** (mean ± SD).",
          f"2. {nm(best)} vs. runner-up {nm(rank[1])} on Macro-F1: {sig_text(p_bw)} (Welch's t-test, Bonferroni α = 0.0083)."]
    small = min(summ.index, key=lambda a: float(summ.loc[a, 'Params (M)']))
    hl.append(f"3. The lightest model, **{nm(small)}** ({summ.loc[small, 'Params (M)']}M params, "
              f"{summ.loc[small, 'Latency (ms/img)']} ms/img), reaches Macro-F1 {summ.loc[small, 'Macro F1 (%)']}%.")
    if strat is not None:
        s = strat[strat.metric == 'test macro-F1'].set_index('arch')
        parts = [f"{nm(a)} {s.loc[a, 'diff LB−FZ (pp) [95% CI]']} pp" for a in s.index]
        hl.append('4. **Fine-tuning the last block vs. a frozen backbone** (Macro-F1 difference, 95% CI): ' + '; '.join(parts) + '.')
    B['highlight'] = '\n'.join(hl)
    B['conclusion'] = '\n'.join([
        f"- We compared 3 pre-trained CNNs (VGG16, ResNet50, MobileNetV2) on a 10-class sport dataset, each tuned with its own W&B Sweep per strategy.",
        f"- Ranking by mean test Macro-F1: " + ' > '.join(f"{nm(a)} ({summ.loc[a, 'Macro F1 (%)']}%)" for a in rank) + '.',
        f"- Best model: **🏆{nm(best)}** ({summ.loc[best, 'strategy']}); {nm(best)} vs. {nm(rank[1])}: {sig_text(p_bw)}.",
        f"- Strategy chosen on validation per model: " + ', '.join(f"{nm(a)} → `{summ.loc[a, 'strategy']}`" for a in rank) + '.'])

    # ---------------- p-values ----------------
    if pv is not None:
        p = pv.copy()
        p['pair'] = p.A.map(nm) + ' vs ' + p.B.map(nm)
        p['diff (pp)'] = p['diff A-B (pp)'].map(lambda v: f'{v:+.2f}')
        p['p-value'] = p['p-value'].map(lambda v: 'n/a' if pd.isna(v) else f'{v:.4f}')
        sigc = [c for c in p.columns if c.startswith('sig @Bonferroni')][0]
        p['significant (Bonferroni)'] = p[sigc].map(lambda v: '✅' if str(v) == 'True' else '❌')
        B['pvalues'] = md_table(p[['metric', 'pair', 'diff (pp)', 'p-value', 'significant (Bonferroni)']], index=False)

    # ---------------- strategy ablation ----------------
    if strat is not None:
        s = strat.copy(); s['arch'] = s.arch.map(nm)
        s = s.rename(columns={'arch': 'model'})
        for c in ['p (Welch)', 'p adj (Bonferroni ×3)', "Cohen's d"]:
            if c in s: s[c] = s[c].map(lambda v: 'n/a' if pd.isna(v) else f'{float(v):.4g}')
        B['strategy'] = md_table(s, index=False) + '\n\n> n/a = undefined because a model gave identical scores for every seed.'

    # ---------------- per-class ----------------
    pc = F('perclass_f1.csv')
    if pc.exists():
        d = pd.read_csv(pc, index_col=0)
        d.columns = [nm(c) for c in d.columns]
        lines = ['**Per-class F1 (%), mean ± SD**', '', md_table(d.rename_axis('Class')), '']
        for a in d.columns:
            lo = d[a].map(mean_of).idxmin()
            lines.append(f'- Weakest class for {a}: **{lo}** ({d.loc[lo, a]}%)')
        B['perclass'] = '\n'.join(lines)

    # ---------------- configs ----------------
    cf = F('cfg_strategy.json')
    if cf.exists():
        c = json.load(open(cf)); chosen = json.load(open(F('best_cfg.json'))) if F('best_cfg.json').exists() else {}
        rows = []
        for a, by_s in c.items():
            for s, v in by_s.items():
                mark = ' ✅' if chosen.get(a, {}).get('strategy') == s else ''
                rows.append({'Model': nm(a), 'Strategy': f'`{s}`{mark}', 'lr': f"{v['lr']:.2e}",
                             'weight_decay': v['weight_decay'], 'hidden (H)': v['hidden'], 'dropout (p)': v['dropout']})
        B['cfg'] = md_table(pd.DataFrame(rows), index=False) + '\n\n> ✅ = strategy selected as the main model (higher mean val macro-F1 over the ' + str(n_seeds) + ' seeds).'

    # ---------------- data ----------------
    sp = F('splits.csv')
    if sp.exists():
        d = pd.read_csv(sp)
        tb = pd.crosstab(d.label, d.split)[['train', 'val', 'test']]
        tb['total'] = tb.sum(axis=1); tb.loc['**Total**'] = tb.sum()
        B['split'] = md_table(tb.rename_axis('Class'))
        cnt = d.label.value_counts()
        B['imbalance'] = (f"After cleaning: **{len(d):,} images**; largest class **{cnt.idxmax()}** ({cnt.max()}), "
                          f"smallest **{cnt.idxmin()}** ({cnt.min()}) → imbalance ratio (max/min) = **{cnt.max() / cnt.min():.2f}**.")
    lk = F('near_duplicate_leaks.csv')
    if lk.exists():
        l = pd.read_csv(lk)
        if l.empty:
            B['leak'] = '> ✅ **No near-duplicate pairs across splits** (dHash distance ≤ 4).'
        else:
            nt = ((l.split_a == 'test') | (l.split_b == 'test')).sum()
            labels = 'all pairs have the same label' if (l.label_a == l.label_b).all() else 'some pairs have conflicting labels'
            n_test = (pd.read_csv(sp).split == 'test').sum() if sp.exists() else 0
            effect = (f', and {nt} of the {n_test:,} test images has a near-copy in another split, so the effect on test '
                      f'accuracy is at most {nt}/{n_test:,} ≈ {100 * nt / n_test:.2f} pp') if n_test else ''
            B['leak'] = (f'> ⚠️ **{len(l)} near-duplicate pair(s) across splits** ({nt} involving the test set). '
                         f'They were **kept** (`REMOVE_NEAR_DUPLICATES = False`) and are reported as a limitation: {labels}{effect}.\n\n'
                         '<p align="center"><img src="outputs/figs/2_near_duplicates.png" width="500"></p>')
    return B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='outputs'); ap.add_argument('--readme', default='README.md')
    ap.add_argument('--n-seeds', type=int, default=5, help='len(SEEDS) used in the notebook')
    a = ap.parse_args()
    blocks = build(Path(a.out), a.n_seeds)
    text = Path(a.readme).read_text(encoding='utf-8')
    for name, body in blocks.items():
        body = re.sub(r'\[[+-]?nan, [+-]?nan\]', '[n/a]', body)
        pat = re.compile(rf'(<!-- AUTO:{name} -->\n).*?(<!-- /AUTO:{name} -->)', re.S)
        if not pat.search(text): print(f'  (no AUTO:{name} block in README, skipped)'); continue
        text = pat.sub(lambda m: m.group(1) + body + '\n' + m.group(2), text)
    Path(a.readme).write_text(text, encoding='utf-8')
    print(f'updated {a.readme}: ' + ', '.join(blocks))


if __name__ == '__main__':
    main()
