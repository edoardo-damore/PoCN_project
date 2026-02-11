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
            "mean": data[:, 2 * i],
            "std": data[:, 2 * i + 1],
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

    fig, ax = plt.subplots()

    offset = np.array(retrieval_time) / 10
    for i, update_rule in enumerate(UpdateRule):
        ax.errorbar(
            retrieval_time + offset * i,
            data[2 * i + 1, 1:],
            yerr=data[2 * i + 2, 1:],
            fmt="o-",
            capsize=2,
            label=f"{update_rule.name}",
        )

    ax.set_ylim(0, 1)
    ax.set_xscale("log")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel("fraction")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(file_path)
    return


def plot_avg_strat_over_degree(
    data_path: Path,
    file_path: Path,
):
    data = pl.read_csv(data_path).to_numpy()

    fig, ax = plt.subplots()

    nonzero = np.sum(data[1:, :] > 0, axis=0)
    nonzero = np.where(nonzero > 0, nonzero, 1)
    x = data[0, :]
    y = np.sum(data[1:, :], axis=0) / nonzero

    x = x[y > 0]
    y = y[y > 0]
    ax.errorbar(
        x,
        y,
        # yerr=np.std(data[1:, :], axis=0),
        fmt="o",
        capsize=2,
    )

    ax.set_ylim(0, 1)
    ax.set_xscale("log")
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$<p>_k$")
    # ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(file_path)
    return
