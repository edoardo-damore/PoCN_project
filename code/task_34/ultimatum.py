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

    from joblib import Parallel, delayed
    from pprint import pprint

    from typing import Literal

    plt.style.use("default")
    return np, nx, plt, random


@app.cell
def _(nx, random):
    # possible update strategies: REP, SP, MOR, UI
    # possible game strategies: EMP, PRG, RND

    # node contents:
    # - proposal
    # - acceptance
    # - payoff
    class UltimatumGame:
        """
        type GameStrategy = Literal["EMP", "PRG", "RND"]
        type UpdateStrategy = Literal["MOR", "REP", "SP", "UI"]
        """

        def __init__(
            self,
            g: nx.Graph,
            update_strategy,#: UpdateStrategy,
            game_strategy,#: GameStrategy,
        ) -> None:
            """
            REP: replicator
            SP: social penalty
            MOR: Moran
            UI: unconditional imitation
            """
            self.g = g
            self.update_strategy = update_strategy
            self.game_strategy = game_strategy
            attributes = self._init_node_attributes()
            nx.set_node_attributes(self.g, attributes)
            return

        # INITIALIZATION METHODS

        def _init_node_attributes(
            self, nodes: list[int] | None = None
        ) -> dict[int, dict[str, float]]:
            if nodes is None:
                nodes = self.g.nodes()

            attributes = {
                node: {"proposal": random.random(), "payoff": 0} for node in nodes
            }

            match self.game_strategy:
                case "EMP":
                    for node in attributes.keys():
                        attributes[node]["acceptance"] = attributes[node][
                            "proposal"
                        ]
                case "PRG":
                    for node in attributes.keys():
                        attributes[node]["acceptance"] = (
                            1 - attributes[node]["proposal"]
                        )
                case "RND":
                    for node in attributes.keys():
                        attributes[node]["acceptance"] = random.random()
            return attributes

        # GAME METHODS

        def _compute_single_payoff(self, i: int, j: int) -> float:
            node_i = self.g.nodes[i]
            node_j = self.g.nodes[i]
            offerer_payoff = (
                1 - node_i["proposal"]
                if node_i["proposal"] <= node_j["acceptance"]
                else 0
            )
            respondent_payoff = (
                node_j["proposal"]
                if node_j["proposal"] <= node_i["acceptance"]
                else 0
            )
            return offerer_payoff + respondent_payoff

        def _game_step(self) -> None:
            for i, j in self.g.edges():
                self.g.nodes[i]["payoff"] += self._compute_single_payoff(i, j)
                self.g.nodes[j]["payoff"] += self._compute_single_payoff(j, i)
            return

        # UPDATE METHODS

        def _REP(self, i: int) -> dict[str, float]:
            j = random.choice(list(self.g.neighbors(i)))
            node_i = self.g.nodes[i]
            node_j = self.g.nodes[j]

            p = (
                (node_j["payoff"] - node_i["payoff"])
                / (2 * max(self.g.degree(i), self.g.degree(j)))
                if node_i["payoff"] < node_j["payoff"]
                else 0
            )

            return node_j if random.random() < p else node_i

        def _UI(self, i: int) -> dict[str, float]:
            j = max(self.g.neighbors(i), key=lambda n: self.g.nodes[n]["payoff"])
            node_i = self.g.nodes[i]
            node_j = self.g.nodes[j]

            return node_j if node_i["payoff"] < node_j["payoff"] else node_i

        def _MOR(self, i: int) -> dict[str, float]:
            neighbors = list(self.g.neighbors(i))
            payoffs = [self.g.nodes[j]["payoff"] for j in neighbors]
            if sum(payoffs) == 0:
                return self.g.nodes[i]

            p = [payoff / sum(payoffs) for payoff in payoffs]

            j = random.choices(neighbors, weights=p)[0]
            return self.g.nodes[j]

        def _SP(self) -> dict[int, dict[str, float]]:
            j = min(self.g.nodes(), key=lambda n: self.g.nodes[n]["payoff"])
            neighbors = list(self.g.neighbors(j)).append(j)
            return self._init_node_attributes(neighbors)

        def _update_step(self) -> None:
            new_state: dict[int, dict[str, float]] = {}
            match self.update_strategy:
                case "REP":
                    for i in self.g.nodes():
                        new_state[i] = self._REP(i)
                case "UI":
                    for i in self.g.nodes():
                        new_state[i] = self._UI(i)
                case "MOR":
                    for i in self.g.nodes():
                        new_state[i] = self._MOR(i)
                case "SP":
                    new_state = self._SP()
            nx.set_node_attributes(self.g, new_state)
            return

        def _reset_payoffs(self) -> None:
            nx.set_node_attributes(self.g, 0, "payoff")
            return

        # RETRIEVAL METHODS

        def get_proposal_distribution(self) -> list[float]:
            return list(nx.get_node_attributes(self.g, "proposal").values())

        def get_acceptance_distribution(self) -> list[float]:
            return list(nx.get_node_attributes(self.g, "acceptance").values())

        def get_payoff_distribution(self) -> list[float]:
            return list(nx.get_node_attributes(self.g, "payoff").values())

        def step(
            self, get_distributions: bool = False
        ) -> tuple[list[float], list[float], list[float]] | None:
            self._reset_payoffs()
            self._game_step()

            if get_distributions:
                proposal_distribution = self.get_proposal_distribution()
                acceptance_distribution = self.get_acceptance_distribution()
                payoff_distribution = self.get_payoff_distribution()

            self._update_step()

            if get_distributions:
                return (
                    proposal_distribution,
                    acceptance_distribution,
                    payoff_distribution,
                )
            return
    return (UltimatumGame,)


@app.cell
def _(nx):
    def generate_sbm(n: int, l: int, p_in: float, p_out: float):
        sizes = [n] * l
        edge_prob = [
            [
                p_in * p_out ** (min(abs(j - i), abs(i + l - j), abs(i - l - j)))
                for j in range(l)
            ]
            for i in range(l)
        ]
        return nx.stochastic_block_model(sizes, edge_prob)
    return (generate_sbm,)


@app.cell
def _(UltimatumGame, generate_sbm):
    # g = nx.barabasi_albert_graph(1000, 1)
    g = generate_sbm(100, 8, 0.5, 0.05)
    ug = UltimatumGame(g, update_strategy="UI", game_strategy="EMP")
    return g, ug


@app.cell
def _(plt, ug):
    d0 = ug.get_proposal_distribution()
    plt.hist(d0, bins=20)
    plt.xlim(0, 1)
    plt.show()
    return


@app.cell
def _(ug):
    for _ in range(1000):
        ug.step()
    return


@app.cell
def _(ug):
    proposal_distribution, acceptance_distribution, payoff_distribution = ug.step(
        get_distributions=True
    )
    return acceptance_distribution, payoff_distribution, proposal_distribution


@app.cell
def _(acceptance_distribution, plt, proposal_distribution):
    plt.hist(
        [proposal_distribution, acceptance_distribution],
        stacked=False,
        bins=20,
        label=["p", "q"],
    )
    plt.xlim(0, 1)
    plt.legend()
    plt.show()
    return


@app.cell
def _(payoff_distribution, plt):
    plt.hist(payoff_distribution, bins=50)
    plt.show()
    return


@app.cell
def _(nx, plt, ug):
    degree_dist = nx.degree_histogram(ug.g)
    plt.bar(range(len(degree_dist)), degree_dist)
    plt.show()
    return


@app.cell
def _(np, payoff_distribution, plt, ug):
    degrees = np.array(list(dict(ug.g.degree()).values()))
    plt.scatter(x=degrees, y=payoff_distribution)
    plt.xlabel("degree")
    plt.ylabel("payoff")
    plt.show()
    return


@app.cell
def _(g, nx, plt, ug):
    nx.draw(
        ug.g,
        pos=nx.spring_layout(ug.g, k=0.5, iterations=200),
        node_size=5,
        width=0.1,
        alpha=0.5,
        node_color=nx.get_node_attributes(g, "proposal").values(),
        cmap="Reds",
    )
    plt.show()
    return


if __name__ == "__main__":
    app.run()
