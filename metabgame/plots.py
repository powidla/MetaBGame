import os
import string
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
from scipy.stats import sem, t as t_dist
from typing import List, Tuple

NASH_COLOR = "#b3b3ff"
BAYESIAN_COLOR = "#ffb3b3"
REWARD_TITLE_SIZE = 26
REWARD_LABEL_SIZE = 28
REWARD_TICK_SIZE = 22
REWARD_LEGEND_SIZE = 24


def mean_ci(data: np.ndarray, conf: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
    n = data.shape[0]
    mean = np.mean(data, axis=0)
    h = sem(data, axis=0) * t_dist.ppf((1 + conf) / 2., n - 1)
    return mean, h


def reward_curves(res: list, acnts: List[int] = None, supd: int = 100, conf: float = 0.95, spath: str = "./scaling_results/reward_scaling.png"):
    if acnts is None:
        acnts = [r["num_agents"] for r in res]
    rord = [(n, next((r for r in res if r["num_agents"] == n), None)) for n in acnts]
    npan = len(rord)
    fig, axes = plt.subplots(1, npan, figsize=(6 * npan, 6))
    if npan == 1:
        axes = [axes]
    ltrs = [f"({l})" for l in string.ascii_lowercase]

    for idx, (nagt, rsl) in enumerate(rord):
        ax = axes[idx]
        if rsl is None:
            ax.set_visible(False)
            continue
        ntrj = np.array(rsl.get("nash_trajectory", []))
        btrj = np.array(rsl.get("bayesian_trajectory", []))
        if ntrj.ndim != 2 or btrj.ndim != 2:
            ax.set_visible(False)
            continue
        xv = np.arange(ntrj.shape[1]) * supd
        mnsh, cnsh = mean_ci(ntrj, conf)
        ax.plot(xv, mnsh, color=NASH_COLOR, linewidth=3, label="Nash")
        ax.fill_between(xv, mnsh - cnsh, mnsh + cnsh, color=NASH_COLOR, alpha=1.0)
        mbay, cbay = mean_ci(btrj, conf)
        ax.plot(xv, mbay, color=BAYESIAN_COLOR, linewidth=3, label="Bayesian")
        ax.fill_between(xv, mbay - cbay, mbay + cbay, color=BAYESIAN_COLOR, alpha=1.0)
        ax.set_title(f"{ltrs[idx]} N = {nagt}", fontsize=REWARD_TITLE_SIZE, pad=20)
        ax.set_xlabel("Steps", fontsize=REWARD_LABEL_SIZE)
        if idx == 0:
            ax.set_ylabel("Total Reward", fontsize=REWARD_LABEL_SIZE)
        if idx == npan - 1:
            ax.legend(fontsize=REWARD_LEGEND_SIZE, loc="best")
        ax.tick_params(axis="both", labelsize=REWARD_TICK_SIZE)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x/1e3:.1f}"))
        ax.text(0.0, 1.02, "1e3", transform=ax.transAxes, fontsize=REWARD_TICK_SIZE)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x/1e6:.1f}"))
        ax.text(1.0, -0.15, "1e6", transform=ax.transAxes, fontsize=REWARD_TICK_SIZE, ha="right")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(spath) or ".", exist_ok=True)
    plt.savefig(spath, dpi=300, bbox_inches="tight")
    plt.close()
    return fig


def scaling_plots(res: list, odir: str = "./scaling_results/"):
    os.makedirs(odir, exist_ok=True)
    acnts = [r["num_agents"] for r in res]
    ttms = [r["time_per_run"] for r in res]
    nrew = [r["nash_mean"] for r in res]
    brew = [r["bayesian_mean"] for r in res]
    advs = [r["bayesian_advantage"] for r in res]
    apct = [r["advantage_percent"] for r in res]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(acnts, ttms, 'o-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Number of Agents', fontsize=12)
    axes[0, 0].set_ylabel('Training Time (seconds)', fontsize=12)
    axes[0, 0].set_title('Computational Scalability', fontsize=14)
    axes[0, 0].set_xscale('log')
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(acnts, nrew, 'o-', label='Nash', linewidth=2, markersize=8)
    axes[0, 1].plot(acnts, brew, 's-', label='Bayesian', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Number of Agents', fontsize=12)
    axes[0, 1].set_ylabel('Mean Total Reward', fontsize=12)
    axes[0, 1].set_title('System Performance Scaling', fontsize=14)
    axes[0, 1].set_xscale('log')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    acol = ['green' if a > 0 else 'red' for a in advs]
    axes[1, 0].bar(range(len(acnts)), advs, color=acol, alpha=0.7)
    axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axes[1, 0].set_xlabel('Number of Agents', fontsize=12)
    axes[1, 0].set_ylabel('Bayesian Advantage', fontsize=12)
    axes[1, 0].set_title('Absolute Advantage', fontsize=14)
    axes[1, 0].set_xticks(range(len(acnts)))
    axes[1, 0].set_xticklabels(acnts, rotation=45)
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    pcol = ['green' if a > 0 else 'red' for a in apct]
    axes[1, 1].bar(range(len(acnts)), apct, color=pcol, alpha=0.7)
    axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axes[1, 1].set_xlabel('Number of Agents', fontsize=12)
    axes[1, 1].set_ylabel('Bayesian Advantage (%)', fontsize=12)
    axes[1, 1].set_title('Relative Advantage', fontsize=14)
    axes[1, 1].set_xticks(range(len(acnts)))
    axes[1, 1].set_xticklabels(acnts, rotation=45)
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f"{odir}/scaling_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()


def behavior_plot(nruns: List[dict], bruns: List[dict], nagt: int, supd: int = 100, cscl: float = 1.0, conf: float = 0.95, spath: str = "./scaling_results/behavior_dynamics.png"):
    os.makedirs(os.path.dirname(spath) or ".", exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(28, 6))
    ltrs = ["(a)", "(b)", "(c)", "(d)"]

    def stack_metric(runs, kbld):
        data = []
        for run in runs:
            tot = None
            for i in range(nagt):
                key = kbld(i)
                if key in run["metrics"]:
                    arr = np.array(run["metrics"][key])
                    tot = arr if tot is None else tot + arr
            if tot is not None:
                data.append(tot)
        return np.stack(data)

    # (a) TOTAL REWARDS
    nrew = stack_metric(nruns, lambda i: f"episode_reward_agent_{i}")
    brew = stack_metric(bruns, lambda i: f"episode_reward_agent_{i}")
    xv = np.arange(nrew.shape[1]) * supd
    mnsh, cnsh = mean_ci(nrew, conf)
    mbay, cbay = mean_ci(brew, conf)
    ax = axes[0]
    ax.plot(xv, mnsh, color=NASH_COLOR, linewidth=3, label="Nash")
    ax.fill_between(xv, mnsh - cnsh, mnsh + cnsh, color=NASH_COLOR, alpha=1.0)
    ax.plot(xv, mbay, color=BAYESIAN_COLOR, linewidth=3, label="Bayesian")
    ax.fill_between(xv, mbay - cbay, mbay + cbay, color=BAYESIAN_COLOR, alpha=1.0)
    ax.set_title(f"{ltrs[0]} Total Rewards", fontsize=REWARD_TITLE_SIZE)
    ax.set_xlabel("Steps", fontsize=REWARD_LABEL_SIZE)
    ax.set_ylabel("Reward", fontsize=REWARD_LABEL_SIZE)
    ax.legend(fontsize=REWARD_LEGEND_SIZE)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=REWARD_TICK_SIZE)

    # (b) COOP VS COMP
    cop = stack_metric(nruns, lambda i: f"agent_{i}_cooperative")
    cmp = stack_metric(nruns, lambda i: f"agent_{i}_competitive")
    cop = cop * cscl
    mcop, ccop = mean_ci(cop, conf)
    mcmp, ccmp = mean_ci(cmp, conf)
    ax = axes[1]
    ax.plot(xv, mcop, color=NASH_COLOR, linewidth=3, label="Coop")
    ax.fill_between(xv, mcop - ccop, mcop + ccop, color=NASH_COLOR, alpha=1.0)
    ax.plot(xv, mcmp, color=BAYESIAN_COLOR, linewidth=3, label="Comp")
    ax.fill_between(xv, mcmp - ccmp, mcmp + ccmp, color=BAYESIAN_COLOR, alpha=1.0)
    ax.set_title(f"{ltrs[1]} Coop vs Comp", fontsize=REWARD_TITLE_SIZE)
    ax.set_xlabel("Steps", fontsize=REWARD_LABEL_SIZE)
    ax.set_ylabel("Actions", fontsize=REWARD_LABEL_SIZE)
    ax.legend(fontsize=REWARD_LEGEND_SIZE)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=REWARD_TICK_SIZE)

    # PRECOMPUTE BAYES ACTIONS
    bcop = stack_metric(bruns, lambda i: f"agent_{i}_cooperative")
    bcmp = stack_metric(bruns, lambda i: f"agent_{i}_competitive")

    # (c) NORMALIZED COOPERATION INDEX
    eps = 1e-8
    nidx = (cop - cmp) / (cop + cmp + eps)
    bidx = (bcop - bcmp) / (bcop + bcmp + eps)
    mnix, cnix = mean_ci(nidx, conf)
    mbix, cbix = mean_ci(bidx, conf)
    ax = axes[2]
    ax.plot(xv, mnix, color=NASH_COLOR, linewidth=3, label="Nash")
    ax.fill_between(xv, mnix - cnix, mnix + cnix, color=NASH_COLOR, alpha=1.0)
    ax.plot(xv, mbix, color=BAYESIAN_COLOR, linewidth=3, label="Bayesian")
    ax.fill_between(xv, mbix - cbix, mbix + cbix, color=BAYESIAN_COLOR, alpha=1.0)
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_ylim([-1.05, 1.05])
    ax.set_title(f"{ltrs[2]} Cooperation Ratio", fontsize=REWARD_TITLE_SIZE)
    ax.set_xlabel("Steps", fontsize=REWARD_LABEL_SIZE)
    ax.set_ylabel("Norm Actions", fontsize=REWARD_LABEL_SIZE)
    ax.legend(fontsize=REWARD_LEGEND_SIZE)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=REWARD_TICK_SIZE)

    # (d) TOTAL ACTIONS
    ntac = cop + cmp
    btac = bcop + bcmp
    mnac, cnac = mean_ci(ntac, conf)
    mbac, cbac = mean_ci(btac, conf)
    ax = axes[3]
    ax.plot(xv, mnac, color=NASH_COLOR, linewidth=3, label="Nash")
    ax.fill_between(xv, mnac - cnac, mnac + cnac, color=NASH_COLOR, alpha=1.0)
    ax.plot(xv, mbac, color=BAYESIAN_COLOR, linewidth=3, label="Bayesian")
    ax.fill_between(xv, mbac - cbac, mbac + cbac, color=BAYESIAN_COLOR, alpha=1.0)
    ax.set_title(f"{ltrs[3]} Total Actions", fontsize=REWARD_TITLE_SIZE)
    ax.set_xlabel("Steps", fontsize=REWARD_LABEL_SIZE)
    ax.set_ylabel("Num Actions", fontsize=REWARD_LABEL_SIZE)
    ax.legend(fontsize=REWARD_LEGEND_SIZE)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=REWARD_TICK_SIZE)

    # Shared formatting
    for idx, ax in enumerate(axes):
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x/1e6:.1f}"))
        ax.text(1.0, -0.15, "1e6", transform=ax.transAxes, fontsize=REWARD_TICK_SIZE, ha="right")
        if idx in [0, 1, 3]:
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f"{y/1e3:.1f}"))
            ax.text(0.0, 1.02, "1e3", transform=ax.transAxes, fontsize=REWARD_TICK_SIZE)

    plt.tight_layout()
    plt.savefig(spath, dpi=300, bbox_inches="tight")
    plt.close()
    return fig
