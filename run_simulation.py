#!/usr/bin/env python3
"""
Government Inertia Simulation — standalone runner.

Usage:
    python run_simulation.py
    python run_simulation.py --cycles 2000 --seed 42 --output results/run1
"""

import argparse
import copy
import sys

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

import gov_inertia as gi


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Run the government inertia simulation.')
    p.add_argument('--cycles', type=int, default=1500,
                   help='Number of simulation cycles (default: 1500)')
    p.add_argument('--seed', type=int, default=None,
                   help='Random seed for reproducibility')
    p.add_argument('--output', type=str, default='output',
                   help='Output file prefix for CSV and PNG (default: output)')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _collapse_ticks(df, ax):
    """Mark collapse events as vertical red lines on an axes."""
    for idx in df.index[df['collapse'] == 1]:
        ax.axvline(idx, color='red', alpha=0.3, linewidth=0.8)


def plot_overview(df, prefix):
    """4×3 grid: population/gov structure, political economy, failure mechanics, budget."""
    fig, axes = plt.subplots(4, 3, figsize=(15, 16))
    fig.suptitle('Government Inertia — Overview', fontsize=15, y=0.99)

    panels = [
        # (column, title, colour, yscale)
        ('N',               'Population (N)',            'steelblue',       'log'),
        ('G',               'Governing Agencies (G)',    'darkorange',      'linear'),
        ('H',               'Bureaucratic Depth (H)',    'firebrick',       'log'),
        ('Lambda_current',  'Legitimacy (Λ)',            'royalblue',       'linear'),
        ('gini',            'Inequality (Gini)',         'sienna',          'linear'),
        ('mu_0',            'Wealth Floor (μ₀)',         'mediumseagreen',  'linear'),
        ('D',               'Divergence (D)',            'purple',          'linear'),
        ('mean_c',          'Mean Compliance (c̄)',       'seagreen',        'linear'),
        ('P_f',             'Failure Probability (Pf)',  'crimson',         'linear'),
        ('C_surveillance',  'Surveillance Cost',         'saddlebrown',     'log'),
        ('C_social',        'Social Enforcement Cost',   'mediumvioletred', 'log'),
        ('R',               'Tax Revenue (R)',           'teal',            'log'),
    ]

    for ax, (col, title, colour, yscale) in zip(axes.flat, panels):
        if col not in df.columns:
            ax.set_visible(False)
            continue
        ax.plot(df.index, df[col], color=colour, linewidth=1.4)
        _collapse_ticks(df, ax)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Cycle', fontsize=8)
        ax.set_yscale(yscale)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=7)

    # Annotate collapse count
    n_collapses = int(df['collapse'].sum())
    fig.text(0.01, 0.01, f'Total collapse events: {n_collapses}',
             fontsize=9, color='red')

    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    path = f'{prefix}_overview.png'
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  saved {path}')


def plot_dynamics(df, prefix):
    """Phase portrait + cost-vs-revenue."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Government Inertia — Dynamics', fontsize=14)

    # --- Phase portrait: D vs mean_c, coloured by cycle ---
    ax = axes[0]
    cycles = df.index.to_numpy()
    colors = cm.plasma(cycles / max(cycles.max(), 1))
    for i in range(len(df) - 1):
        ax.plot(df['D'].iloc[i:i+2], df['mean_c'].iloc[i:i+2],
                color=colors[i], linewidth=1.2)
    sc = ax.scatter(df['D'], df['mean_c'], c=cycles, cmap='plasma', s=4, zorder=5)
    ax.scatter(df['D'].iloc[0], df['mean_c'].iloc[0],
               color='lime', s=80, zorder=10, label='Start')
    ax.scatter(df['D'].iloc[-1], df['mean_c'].iloc[-1],
               color='red', s=80, zorder=10, label='End')
    fig.colorbar(sc, ax=ax, label='Cycle')
    D_crit = df['D_crit'].iloc[0] if 'D_crit' in df.columns else 0.3
    ax.axvline(D_crit, color='red', linestyle='--', linewidth=1, label='D_crit')
    ax.set_xlabel('Divergence (D)')
    ax.set_ylabel('Mean Compliance (c̄)')
    ax.set_title('Phase Portrait: D vs Compliance')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    # --- Revenue vs total cost (log scale) ---
    ax = axes[1]
    ax.plot(df.index, df['R'], color='teal', linewidth=1.4, label='Revenue R')
    if 'C_total_computed' in df.columns:
        ax.plot(df.index, df['C_total_computed'], color='firebrick',
                linewidth=1.4, label='Total Cost C')
    if 'C_social' in df.columns:
        ax.plot(df.index, df['C_social'], color='mediumvioletred',
                linewidth=1.0, linestyle='--', label='C_social')
    if 'C_surveillance' in df.columns:
        ax.plot(df.index, df['C_surveillance'], color='saddlebrown',
                linewidth=1.0, linestyle=':', label='C_surveillance')
    _collapse_ticks(df, ax)
    ax.set_xlabel('Cycle')
    ax.set_ylabel('Value')
    ax.set_title('Revenue vs. Costs')
    ax.set_yscale('log')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    path = f'{prefix}_dynamics.png'
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  saved {path}')


def plot_political_economy(df, prefix):
    """Legitimacy, inequality, and reform/collapse events."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Government Inertia — Political Economy', fontsize=13)

    # Legitimacy over time
    ax = axes[0]
    ax.fill_between(df.index, df['Lambda_current'], alpha=0.2, color='royalblue')
    ax.plot(df.index, df['Lambda_current'], color='royalblue', linewidth=1.4)
    _collapse_ticks(df, ax)
    for idx in df.index[df['reform'] == 1]:
        ax.axvline(idx, color='green', alpha=0.4, linewidth=0.8)
    ax.set_title('Legitimacy (Λ)\n[red=collapse, green=reform]')
    ax.set_xlabel('Cycle')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.25)

    # Gini over time
    ax = axes[1]
    ax.plot(df.index, df['gini'], color='sienna', linewidth=1.4)
    ax.set_title('Inequality (Gini)')
    ax.set_xlabel('Cycle')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.25)

    # Cumulative events
    ax = axes[2]
    ax.step(df.index, df['collapse'].cumsum(), color='crimson', linewidth=1.5, label='Collapses')
    ax.step(df.index, df['reform'].cumsum(), color='seagreen', linewidth=1.5, label='Reforms')
    if 'insolvent' in df.columns:
        ax.step(df.index, df['insolvent'].cumsum(), color='darkorange',
                linewidth=1.5, linestyle='--', label='Insolvencies')
    ax.set_title('Cumulative Events')
    ax.set_xlabel('Cycle')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    path = f'{prefix}_political_economy.png'
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  saved {path}')


def plot_collapse_timeline(df, prefix):
    """P_f and cumulative collapse events."""
    fig, ax1 = plt.subplots(figsize=(10, 4))
    fig.suptitle('Collapse Timeline', fontsize=13)

    ax1.fill_between(df.index, df['P_f'], alpha=0.25, color='crimson', label='P_f')
    ax1.plot(df.index, df['P_f'], color='crimson', linewidth=1.2)
    ax1.axhline(0.5, color='crimson', linestyle='--', linewidth=0.8, alpha=0.6)
    ax1.set_ylabel('Failure Probability (P_f)', color='crimson')
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis='y', colors='crimson')

    ax2 = ax1.twinx()
    ax2.step(df.index, df['collapse'].cumsum(), color='navy', linewidth=1.5,
             label='Cumulative collapses')
    ax2.set_ylabel('Cumulative Collapse Events', color='navy')
    ax2.tick_params(axis='y', colors='navy')

    ax1.set_xlabel('Cycle')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.2)

    plt.tight_layout()
    path = f'{prefix}_collapse.png'
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  saved {path}')


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(history):
    initial = history[0]
    final = history[-1]
    n = len(history) - 1
    print(f'\n{"="*55}')
    print(f'  Simulation summary  ({n} cycles)')
    print(f'{"="*55}')
    rows = [
        ('Population N',        initial['N'],          final['N'],          '.3e'),
        ('Gov agencies G',      initial['G'],          final['G'],          '.2f'),
        ('Hierarchy H',         initial['H'],          final['H'],          '.3e'),
        ('Legitimacy Λ',        initial['Lambda'],     final['Lambda'],     '.4f'),
        ('Divergence D',        initial['D'],          final['D'],          '.4f'),
        ('Mean compliance c̄',  initial['mean_c'],     final['mean_c'],     '.4f'),
        ('Failure prob P_f',    initial['P_f'],        final['P_f'],        '.4f'),
        ('Inequality (Gini)',   initial['gini'],       final['gini'],       '.4f'),
        ('Wealth floor μ₀',     initial['mu_0'],       final['mu_0'],       '.4f'),
        ('Stolen funds',        initial['stolen_funds'], final['stolen_funds'], '.3e'),
        ('Savings',             initial['savings'],    final['savings'],    '.3e'),
    ]
    for label, v0, vf, fmt in rows:
        print(f'  {label:<24} {v0:{fmt}}  →  {vf:{fmt}}')
    print(f'\n  Collapse events:  {int(final["collapse_events"])}')
    print(f'  Reform events:    {int(final["reform_events"])}')
    print(f'  Insolvency events:{int(final["insolvent_count"])}')
    print(f'{"="*55}\n')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        print(f'Random seed: {args.seed}')

    print(f'Running {args.cycles} cycles...')
    params = copy.deepcopy(gi.initial_conditions)
    history = gi.run_simulation(params, args.cycles)

    df = pd.DataFrame(history)

    csv_path = f'{args.output}.csv'
    df.to_csv(csv_path, index=False)
    print(f'  saved {csv_path}')

    print('Generating plots...')
    plot_overview(df, args.output)
    plot_dynamics(df, args.output)
    plot_political_economy(df, args.output)
    plot_collapse_timeline(df, args.output)

    print_summary(history)


if __name__ == '__main__':
    main()
