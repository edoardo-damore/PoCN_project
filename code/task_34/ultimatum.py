import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    import graph_tool as gt
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    import random

    from typing import Literal

    plt.style.use("default")
    return Literal, nx, random


@app.cell
def _(nx):
    g = nx.barabasi_albert_graph(100, 1)
    return (g,)


@app.cell
def _(Literal, nx, random):
    def generate_strategies(
        g: nx.Graph,
        strategy: Literal["empathetic", "pragmatic", "random"] = "random",
    ):
        if strategy not in ["empathetic", "pragmatic", "random"]:
            raise ValueError("strategy value not valid")

        proposals = {node: random.random() for node in g.nodes}
        match strategy:
            case "empathetic":
                nx.set_node_attributes(g, proposals, "proposal")
                nx.set_node_attributes(g, proposals, "acceptance")
            case "pragmatic":
                nx.set_node_attributes(g, proposals, "proposal")
                nx.set_node_attributes(
                    g,
                    {node: 1 - proposal for node, proposal in proposals.items()},
                    "acceptance",
                )
            case "random":
                nx.set_node_attributes(g, proposals, "proposal")
                nx.set_node_attributes(
                    g, {node: random.random() for node in g.nodes}, "acceptance"
                )
        return
    return (generate_strategies,)


@app.cell
def _(g, generate_strategies):
    generate_strategies(g, "pragmatic")
    return


@app.cell
def _(g):
    g.nodes[0]
    return


if __name__ == "__main__":
    app.run()
