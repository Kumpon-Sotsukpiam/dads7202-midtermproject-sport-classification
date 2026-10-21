"""Redraw the figures that compare models, from the CSVs in outputs/ (no re-training, notebook untouched).

Usage (from the repo root):
    python replot_figures.py            # overwrites outputs/figs/6_perclass_f1.png and 7x_strategy_diff.png

Every model uses the same colour and display name in every figure (same colours as COLORS in the notebook).
Note: re-running the notebook overwrites these two PNGs with its own versions; run this script again afterwards.
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

OUT = Path('outputs')
ARCHS = ['vgg16', 'mobilenet_v2', 'resnet50']
NAMES = {'vgg16': 'VGG16', 'mobilenet_v2': 'MobileNetV2', 'resnet50': 'ResNet50'}
COLORS = {'vgg16': 'tab:blue', 'mobilenet_v2': 'tab:green', 'resnet50': 'tab:red'}


def mean_sd(s):                                   # "87.29 ± 1.58" -> (87.29, 1.58)
    m, sd = str(s).split('±')
    return float(m), float(sd)


def savefig(fig, name):
    fig.savefig(OUT / 'figs' / f'{name}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('saved', OUT / 'figs' / f'{name}.png')


def perclass_f1():
    d = pd.read_csv(OUT / 'perclass_f1.csv', index_col=0)
    classes, w = list(d.index), 0.27
    fig, ax = plt.subplots(figsize=(13, 4.5))
    lo = 100
    for j, a in enumerate(ARCHS):
        m, sd = zip(*d[a].map(mean_sd))
        ax.bar([i + (j - 1) * w for i in range(len(classes))], m, w, yerr=sd, capsize=3,
               label=NAMES[a], color=COLORS[a])
        lo = min(lo, min(x - s for x, s in zip(m, sd)))
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=30)
    ax.set_ylim(max(0, (lo // 5) * 5 - 5), 100)     # zoom in: all F1 values are well above 80%
    ax.set_ylabel('F1 (%)'); ax.grid(axis='y', alpha=.3)
    ax.set_title('Per-class test F1 (mean ± SD over 5 seeds); y-axis does not start at 0')
    ax.legend(ncol=3, loc='lower center', bbox_to_anchor=(0.5, 1.08), frameon=False)
    savefig(fig, '6_perclass_f1')


def strategy_diff():
    s = pd.read_csv(OUT / 'strategy_comparison.csv')
    s = s[s.metric == 'test macro-F1'].set_index('arch')
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    for i, a in enumerate(ARCHS):
        diff, lo, hi = map(float, re.findall(r'[+-]?\d+\.\d+', s.loc[a, 'diff LB−FZ (pp) [95% CI]']))
        ax.errorbar(diff, i, xerr=[[diff - lo], [hi - diff]], fmt='o', capsize=4, color=COLORS[a])
        ax.annotate(f"{diff:+.2f} pp, p_adj = {float(s.loc[a, 'p adj (Bonferroni ×3)']):.3g}", (hi, i),
                    xytext=(6, 0), textcoords='offset points', va='center', fontsize=8)
    ax.axvline(0, c='k', lw=.8); ax.set_yticks(range(len(ARCHS))); ax.set_yticklabels([NAMES[a] for a in ARCHS])
    ax.set_xlim(right=ax.get_xlim()[1] + 1.5)        # room for the labels
    ax.set_xlabel('test macro-F1 difference, last_block − frozen (pp), 95% CI'); ax.grid(alpha=.3)
    savefig(fig, '7x_strategy_diff')


if __name__ == '__main__':
    perclass_f1()
    strategy_diff()
