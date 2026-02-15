import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from pathlib import Path


def plot_degree_distributions(
    graphs: list[nx.Graph],
    labels: list[str],
    save_dir: Path,
) -> None:
    fig, ax = plt.subplots()

    for i, g in enumerate(graphs):
        counts = np.bincount(list(dict(g.degree()).values()))
        d = np.arange(counts.shape[0])
        d, counts = d[counts > 0], counts[counts > 0] / np.sum(counts)

        ax.scatter(d, counts, label=labels[i])

    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$P(k)$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(save_dir / "degree_distributions.png")

    return


def plot_closeness(
    graphs: list[nx.Graph],
    labels: list[str],
    save_dir: Path,
) -> None:
    fig, axs = plt.subplots(ncols=len(graphs), figsize=(5 * len(graphs), 5))

    for i, g in enumerate(graphs):
        pos = {node: (data["lon"], data["lat"]) for node, data in g.nodes(data=True)}
        closeness = list(nx.closeness_centrality(g).values())
        p = nx.draw(
            g,
            pos=pos,
            ax=axs[i],
            node_color=closeness,
            node_size=2,
            width=0,
            arrowstyle="-",
        )
        axs[i].set_title(labels[i])

    fig.colorbar(p, ax=axs[-1], label="Closeness centrality")
    fig.tight_layout()
    fig.savefig(save_dir / "closeness.png")
    return


def plot_betweenness(
    graphs: list[nx.Graph],
    labels: list[str],
    save_dir: Path,
) -> None:
    fig, axs = plt.subplots(ncols=len(graphs), figsize=(5 * len(graphs), 5))

    for i, g in enumerate(graphs):
        pos = {node: (data["lon"], data["lat"]) for node, data in g.nodes(data=True)}
        betweenness = list(nx.betweenness_centrality(g).values())
        p = nx.draw(
            g,
            pos=pos,
            ax=axs[i],
            node_color=betweenness,
            node_size=2,
            width=0,
            arrowstyle="-",
        )
        axs[i].set_title(labels[i])

    fig.colorbar(p, ax=axs[-1], label="Betweenness centrality")
    fig.tight_layout()
    fig.savefig(save_dir / "betweenness.png")
    return


def plot_average_degree_connectivity(
    graphs: list[nx.Graph],
    labels: list[str],
    save_dir: Path,
) -> None:
    fig, ax = plt.subplots()
    for i, g in enumerate(graphs):
        temp = nx.average_degree_connectivity(g)
        degrees = list(temp.keys())
        adc = list(temp.values())
        ax.scatter(degrees, adc, label=labels[i])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$k^{(nn)}$")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(save_dir / "adc.png")
    return
