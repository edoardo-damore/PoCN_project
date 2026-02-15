import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import polars as pl

from pathlib import Path

from visualization import (
    plot_degree_distributions,
    plot_closeness,
    plot_betweenness,
    plot_average_degree_connectivity,
)

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 18,
        "text.latex.preamble": "\\usepackage{amsmath}",
    }
)


def main():
    BASE_DIR = Path(__file__).parent.parent.parent
    DATA_DIR = BASE_DIR / "data" / "task_42"
    FIGURE_DIR = Path("figures")
    if not FIGURE_DIR.exists():
        FIGURE_DIR.mkdir()

    cities = ["London", "Birmingham", "Liverpool"]

    graphs = []

    for city in cities:
        nodes = pl.read_csv(DATA_DIR / city / "nodes.csv")
        edges = pl.read_csv(DATA_DIR / city / "edges.csv")

        g = nx.from_pandas_edgelist(
            edges,
            source="ori_nodeid",
            target="des_nodeid",
            edge_attr=True,
            create_using=nx.DiGraph(),
        )

        g.add_nodes_from(
            (row["nodeid"], row.drop("nodeid").to_dict())
            for _, row in nodes.to_pandas().iterrows()
        )
        graphs.append(g)

    print("assortativity [[out-out, in-out], [out-in, in-in]]")
    for i, g in enumerate(graphs):
        print(cities[i])
        assortativity = [
            [nx.degree_assortativity_coefficient(g, x=a, y=b) for b in ["out", "in"]]
            for a in ["out", "in"]
        ]
        print(assortativity)

    plot_degree_distributions(graphs, cities, FIGURE_DIR)
    # plot_closeness(graphs, cities, FIGURE_DIR)
    # plot_betweenness(graphs, cities, FIGURE_DIR)
    plot_average_degree_connectivity(graphs, cities, FIGURE_DIR)

    return


if __name__ == "__main__":
    main()
