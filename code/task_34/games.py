import networkx as nx
import random

from typing import Any, Literal


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
        if update_rule_distribution is not None and sum(update_rule_distribution) != 1:
            update_rule_distribution = [
                p / sum(update_rule_distribution) for p in update_rule_distribution
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


class WeakPrisonerDilemmaGame:
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

        if update_rule_distribution is not None and sum(update_rule_distribution) != 1:
            update_rule_distribution = [
                p / sum(update_rule_distribution) for p in update_rule_distribution
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
