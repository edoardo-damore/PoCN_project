import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import polars as pl

from pathlib import Path
from typing import Literal

from custom_types import UGStrategy, WPDStrategy, UpdateRule, ResultsDict


def plot_strategy_distribution(
    data_path: Path,
    file_path: Path,
    quantity: Literal["p", "q"],
):
    data = pl.read_csv(data_path).to_numpy()
    bin_edges = data[:, 0]
    retrieval_time = data[0, 1:]

    N = np.sum(data[1:, 1]) * (bin_edges[1] - bin_edges[0])

    fig, ax = plt.subplots()
    for j, t in enumerate(retrieval_time):
        ax.errorbar(
            (bin_edges[1:] + bin_edges[:-1]) / 2,
            data[1:, j + 1] / N,
            fmt="o-",
            label=f"t={int(t)}",
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(
        0,
    )
    ax.set_xlabel(rf"${quantity}$")
    ax.set_ylabel(rf"$D({quantity})$")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(file_path)
    return


def plot_strategy_space(
    data_path: Path,
    file_path: Path,
    scatter_label: str = "nets",
):
    data = pl.read_csv(data_path).to_numpy()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(
        data[:, 0],
        data[:, 1],
        label=scatter_label,
    )
    ax.plot([0, 1], [0, 1], color="r", label=r"$p = q$")
    ax.set_xlabel(r"$p$")
    ax.set_ylabel(r"$q$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(file_path)
    return


def plot_strategy_frequency(
    data_path: Path,
    file_path: Path,
    max_strategies: int = 10,
):
    data = pl.read_csv(data_path).to_numpy()
    strat_freqencies = {
        game_strat.name: {
            "mean": data[:, i],
            "std": data[:, i + 1],
        }
        for i, game_strat in enumerate(UGStrategy)
    }
    fig, ax = plt.subplots()
    mult = -1
    width = 0.25
    for game_strat, counts in strat_freqencies.items():
        offset = width * mult
        ax.bar(
            np.arange(max_strategies) + offset,
            counts["mean"],
            yerr=counts["std"],
            width=width,
            edgecolor="black",
            capsize=2,
            label=game_strat,
        )
        mult += 1
    ax.set_xticklabels([])
    ax.set_xticks(np.arange(max_strategies))
    ax.set_ylabel("average fraction over all nodes")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(file_path)

    return


def plot_update_rule_distribution(
    data_path: Path,
    file_path: Path,
):
    data = pl.read_csv(data_path).to_numpy()
    retrieval_time = data[0, 1:]
    N = np.sum(data[1:, 1])

    fig, ax = plt.subplots()

    for i, update_rule in enumerate(UpdateRule):
        ax.errorbar(
            retrieval_time, data[i + 1, 1:] / N, fmt="o-", label=f"{update_rule.name}"
        )

    ax.set_xscale("log")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel("fraction")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(file_path)
    return


"""
    width = 0.8 / len(retrieval_time)
    mult = -(len(retrieval_time) - 1) / 2
    for i, t in enumerate(retrieval_time):
        offset = mult * width
        ax.bar(
            np.arange(len(UpdateRule)) + offset,
            data[1:, i + 1] / N,
            width=width,
            edgecolor="black",
            capsize=2,
            label=rf"$t={int(t)}$",
        )
        mult += 1
    ax.set_xticks(np.arange(len(UpdateRule)))
    ax.set_xticklabels([update_rule.name for update_rule in UpdateRule])
    # ax.set_xticks([update_rule.name for update_rule in UpdateRule])
    # ax.set_ylabel("average fraction over all nodes")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    """
