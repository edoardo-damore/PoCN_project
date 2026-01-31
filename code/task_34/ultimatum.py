import marimo

__generated_with = "0.19.7"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    import graph_tool as gt
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    import polars as pl
    import random

    from joblib import Parallel, delayed
    from pprint import pprint

    from typing import Literal, Any

    plt.style.use("default")
    return Any, Literal, Parallel, delayed, np, nx, plt, random


@app.cell
def _(Any, Literal, nx, random):
    # possible update strategies: REP, SP, MOR, UI
    # possible game strategies: EMP, PRG, RND

    # node contents:
    # - p
    # - q
    # - payoff
    # - update_rule
    class UltimatumGame:
        def __init__(
            self,
            g: nx.Graph,
            game_strategy: Literal["EMP", "PRG", "RND"] = "RND",
            update_rule_distribution: list[float] | None = None,
        ) -> None:
            if game_strategy not in ["EMP", "PRG", "RND"]:
                raise ValueError("game_strategy must be one of: EMP, PRG or RND")

            # normalizing distributions
            if (
                update_rule_distribution is not None
                and sum(update_rule_distribution) != 1
            ):
                update_rule_distribution = [
                    p / sum(update_rule_distribution)
                    for p in update_rule_distribution
                ]
            self.g = g
            self.game_strategy = game_strategy
            self.update_rule_distribution = update_rule_distribution
            attributes = self._init_node_attributes()
            nx.set_node_attributes(self.g, attributes)
            return

        # INITIALIZATION METHODS

        def _init_node_attributes(
            self, nodes: list[int] | None = None
        ) -> dict[int, dict[str, Any]]:
            if nodes is None:
                nodes = self.g.nodes()

            attributes = {
                node: {
                    "p": random.random(),
                    "payoff": 0,
                    "update_rule": random.choices(
                        ["REP", "UI", "MOR"], self.update_rule_distribution
                    )[0],  # random.choices returns a list, even for just 1 element
                }
                for node in nodes
            }

            match self.game_strategy:
                case "EMP":
                    for node in attributes.keys():
                        attributes[node]["q"] = attributes[node]["p"]
                case "PRG":
                    for node in attributes.keys():
                        attributes[node]["q"] = 1 - attributes[node]["p"]
                case "RND":
                    for node in attributes.keys():
                        attributes[node]["q"] = random.random()
            return attributes

        # GAME METHODS

        def _compute_single_payoff(self, i: int, j: int) -> float:
            node_i = self.g.nodes[i]
            node_j = self.g.nodes[i]
            # i offers to j
            offerer_payoff = 1 - node_i["p"] if node_i["p"] <= node_j["q"] else 0
            # j offers to i
            respondent_payoff = node_j["p"] if node_j["p"] <= node_i["q"] else 0
            return offerer_payoff + respondent_payoff

        def _game_step(self) -> None:
            for i, j in self.g.edges():
                self.g.nodes[i]["payoff"] += self._compute_single_payoff(i, j)
                self.g.nodes[j]["payoff"] += self._compute_single_payoff(j, i)
            return

        # UPDATE METHODS

        def _REP(self, i: int) -> dict[str, Any]:
            j = random.choice(list(self.g.neighbors(i)))
            node_i = self.g.nodes[i]
            node_j = self.g.nodes[j]

            prob = 0
            payoff_diff = node_j["payoff"] - node_i["payoff"]
            if payoff_diff > 0:
                prob = payoff_diff / (2 * max(self.g.degree(i), self.g.degree(j)))

            return node_j if random.random() < prob else node_i

        def _UI(self, i: int) -> dict[str, Any]:
            # argmax over the payoffs
            j = max(self.g.neighbors(i), key=lambda n: self.g.nodes[n]["payoff"])
            node_i = self.g.nodes[i]
            node_j = self.g.nodes[j]

            return node_j if node_i["payoff"] < node_j["payoff"] else node_i

        def _MOR(self, i: int) -> dict[str, Any]:
            neighbors = list(self.g.neighbors(i))
            payoffs = [self.g.nodes[j]["payoff"] for j in neighbors]
            # avoiding division by 0
            if sum(payoffs) < 1e-6:
                return self.g.nodes[i]

            p = [payoff / sum(payoffs) for payoff in payoffs]

            j = random.choices(neighbors, weights=p)[0]
            return self.g.nodes[j]

        def _SP(self) -> dict[int, dict[str, Any]]:
            # argmin over the payoffs
            j = min(self.g.nodes(), key=lambda n: self.g.nodes[n]["payoff"])
            neighbors = list(self.g.neighbors(j)).append(j)
            return self._init_node_attributes(neighbors)

        def _update_step(self) -> None:
            new_state: dict[int, dict[str, Any]] = {}
            for i in self.g.nodes():
                node_i = self.g.nodes[i]
                match node_i["update_rule"]:
                    case "REP":
                        new_state[i] = self._REP(i)
                    case "UI":
                        new_state[i] = self._UI(i)
                    case "MOR":
                        new_state[i] = self._MOR(i)
            nx.set_node_attributes(self.g, new_state)
            return

        def _reset_payoff(self) -> None:
            nx.set_node_attributes(self.g, 0, "payoff")
            return

        # RETRIEVAL METHODS

        def get_p(self) -> list[float]:
            return list(nx.get_node_attributes(self.g, "p").values())

        def get_q(self) -> list[float]:
            return list(nx.get_node_attributes(self.g, "q").values())

        def get_payoff(self) -> list[float]:
            return list(nx.get_node_attributes(self.g, "payoff").values())

        def get_update_rule(self) -> list[str]:
            return list(nx.get_node_attributes(self.g, "update_rule").values())

        # SINGLE STEP (GAME + UPDATE)

        def step(
            self, get_distributions: bool = False
        ) -> tuple[list[float], list[float], list[float], list[str]] | None:
            self._reset_payoff()
            self._game_step()

            # retrieval step happens before update
            # otherwise payoffs would be copied
            if get_distributions:
                p = self.get_p()
                q = self.get_q()
                payoff = self.get_payoff()
                update_rule = self.get_update_rule()

            self._update_step()

            if get_distributions:
                return p, q, payoff, update_rule
            return
    return (UltimatumGame,)


@app.cell
def _(Any, nx, random):
    # possible update strategies: REP, MOR, UI
    # possible game strategies: 0 (collaborate), 1 (defect)

    # node contents:
    # - action
    # - payoff
    # - update_rule
    class WeakPrisonerDilemma:
        def __init__(
            self,
            g: nx.Graph,
            b: float,
            action_distribution: list[float] | None = None,
            update_rule_distribution: list[float] | None = None,
        ) -> None:
            # normalizing distributions
            if action_distribution is not None and sum(action_distribution) != 1:
                action_distribution = [
                    p / sum(action_distribution) for p in action_distribution
                ]

            if (
                update_rule_distribution is not None
                and sum(update_rule_distribution) != 1
            ):
                update_rule_distribution = [
                    p / sum(update_rule_distribution)
                    for p in update_rule_distribution
                ]

            self.g = g
            self.reward = [[1, 0], [b, 0]]
            self.action_distribution = action_distribution
            self.update_rule_distribution = update_rule_distribution
            attributes = self._init_node_attributes()
            nx.set_node_attributes(self.g, attributes)
            return

        # INITIALIZATION METHODS

        def _init_node_attributes(
            self, nodes: list[int] | None = None
        ) -> dict[int, dict[str, Any]]:
            if nodes is None:
                nodes = self.g.nodes()

            attributes = {
                node: {
                    "action": random.choices([0, 1], self.action_distribution)[0],
                    "payoff": 0,
                    "update_rule": random.choices(
                        ["REP", "UI", "MOR"], self.update_rule_distribution
                    )[0],  # random.choices returns a list, even for just 1 element
                }
                for node in nodes
            }
            return attributes

        # GAME METHODS

        def _compute_single_payoff(self, i: int, j: int) -> float:
            node_i = self.g.nodes[i]
            node_j = self.g.nodes[j]
            return self.reward[node_i["action"]][node_j["action"]]

        def _game_step(self) -> None:
            for i, j in self.g.edges():
                self.g.nodes[i]["payoff"] += self._compute_single_payoff(i, j)
                self.g.nodes[j]["payoff"] += self._compute_single_payoff(j, i)
            return

        # UPDATE METHODS

        def _REP(self, i: int) -> dict[str, Any]:
            j = random.choice(list(self.g.neighbors(i)))
            node_i = self.g.nodes[i]
            node_j = self.g.nodes[j]

            prob = 0
            payoff_diff = node_j["payoff"] - node_i["payoff"]
            if payoff_diff > 0:
                prob = payoff_diff / (2 * max(self.g.degree(i), self.g.degree(j)))

            return node_j if random.random() < prob else node_i

        def _UI(self, i: int) -> dict[str, Any]:
            # argmax over the payoffs
            j = max(self.g.neighbors(i), key=lambda n: self.g.nodes[n]["payoff"])
            node_i = self.g.nodes[i]
            node_j = self.g.nodes[j]

            return node_j if node_i["payoff"] < node_j["payoff"] else node_i

        def _MOR(self, i: int) -> dict[str, Any]:
            neighbors = list(self.g.neighbors(i))
            payoffs = [self.g.nodes[j]["payoff"] for j in neighbors]
            # avoiding division by 0
            if sum(payoffs) < 1e-6:
                return self.g.nodes[i]

            p = [payoff / sum(payoffs) for payoff in payoffs]

            j = random.choices(neighbors, weights=p)[0]
            return self.g.nodes[j]

        def _update_step(self) -> None:
            new_state: dict[int, dict[str, Any]] = {}
            for i in self.g.nodes():
                node_i = self.g.nodes[i]
                match node_i["update_rule"]:
                    case "REP":
                        new_state[i] = self._REP(i)
                    case "UI":
                        new_state[i] = self._UI(i)
                    case "MOR":
                        new_state[i] = self._MOR(i)
            nx.set_node_attributes(self.g, new_state)
            return

        def _reset_payoff(self) -> None:
            nx.set_node_attributes(self.g, 0, "payoff")
            return

        # RETRIEVAL METHODS

        def get_action(self) -> list[int]:
            return list(nx.get_node_attributes(self.g, "action").values())

        def get_payoff(self) -> list[float]:
            return list(nx.get_node_attributes(self.g, "payoff").values())

        def get_update_rule(self) -> list[str]:
            return list(nx.get_node_attributes(self.g, "update_rule").values())

        # SINGLE STEP (GAME + UPDATE)

        def step(
            self, get_distributions: bool = False
        ) -> tuple[list[int], list[float], list[str]] | None:
            self._reset_payoff()
            self._game_step()

            # retrieval step happens before update
            # otherwise payoffs would be copied
            if get_distributions:
                action = self.get_action()
                payoff = self.get_payoff()
                update_rule = self.get_update_rule()

            self._update_step()

            if get_distributions:
                return action, payoff, update_rule
            return
    return


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
def _(np, nx):
    def generate_hierarchical_sbm(
        n: int, l: int, p_in: float, p_out: float
    ) -> nx.Graph:
        if np.log2(l) % 1 != 0:
            l = int(2 ** np.floor(np.log2(l)))
            print(f"l will be rounded down to the nearest power of 2: {l}")
        sizes = [n] * l
        edge_prob = p_in * np.ones(shape=(l, l))
        div = 1
        while div < l:
            mult = p_out * np.ones(shape=(l, l))
            for i in range(l // div):
                mult[i * div : (i + 1) * div, i * div : (i + 1) * div] = 1
            edge_prob *= mult
            div *= 2
        print(edge_prob)
        return nx.stochastic_block_model(sizes, edge_prob)
    return (generate_hierarchical_sbm,)


@app.cell
def _(generate_hierarchical_sbm):
    prova = generate_hierarchical_sbm(100, 38, 0.5, 0.05)
    return (prova,)


@app.cell
def _(UltimatumGame, generate_sbm):
    def full_analysis(N: int, epochs: int):
        g = generate_sbm(N // 8, 8, p_in=0.5, p_out=0.05)
        ug = UltimatumGame(
            g, game_strategy="EMP", update_rule_distribution=[0, 0.01, 0.99]
        )
        for _ in range(epochs):
            ug.step()
        p, q, payoff, update_rule = ug.step(get_distributions=True)
        return update_rule
    return (full_analysis,)


@app.cell
def _(Parallel, delayed, full_analysis):
    N = 1000
    epochs = 1000
    p = Parallel(n_jobs=-1)(
        delayed(lambda x: full_analysis(N, epochs))(i) for i in range(10)
    )
    return (p,)


@app.cell
def _(p, plt):
    plt.hist(p, stacked=True)
    plt.show()
    return


@app.cell
def _(nx, plt, prova):
    nx.draw(
        prova,
        pos=nx.spring_layout(prova, k=0.5, iterations=200),
        node_size=5,
        width=0.1,
        alpha=0.5,
        # node_color=nx.get_node_attributes(g, "proposal").values(),
        # cmap="Reds",
    )
    plt.show()
    return


if __name__ == "__main__":
    app.run()
