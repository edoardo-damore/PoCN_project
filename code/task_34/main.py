import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random

from joblib import Parallel, delayed
from pathlib import Path
from typing import Literal, Any, Callable

from analysis import (
    plot_strategy_distribution,
    plot_strategy_space,
    plot_strategy_frequency,
    plot_update_rule_distribution,
)
from custom_types import UGStrategy, WPDStrategy, UpdateRule, ResultsDict
from games import (
    UltimatumGame,
    WeakPrisonerDilemmaGame,
)
from generators import (
    generate_barabasi_albert,
    generate_erdos_renyi,
    generate_hierarchical_sbm,
    generate_watts_strogatz,
)
from utils import (
    save_ug_strat_distribution,
    save_degree_of_selection,
    save_strategy_space,
    save_ug_update_rule_distribution,
)

plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.size": 18})


def run_single_ultimatum_game(
    generator: Callable[[], nx.Graph],
    game_strategy: UGStrategy,
    iterations: int,
    retrieval_time: list[int],
    update_rule_distribution: list[float] | None = None,
) -> dict[str, Any]:
    g = generator()
    ug = UltimatumGame(g, game_strategy, update_rule_distribution)
    results = {}
    for i in range(iterations):
        if i + 1 in retrieval_time:
            p, q, payoff, update_rule = ug.step(get_distributions=True)

            results[i + 1] = {
                "strategy": np.concat(
                    [np.array(p).reshape(-1, 1), np.array(q).reshape(-1, 1)], axis=1
                ),
                "payoff": sum(payoff),
                "update_rule": np.array(update_rule),
                "degree": list(dict(ug.g.degree()).values()),
            }
        else:
            ug.step()
    return results


def run_single_wpd_game(
    generator: Callable[[Any], nx.Graph],
    b: float,
    iterations: int,
    retrieval_time: list[int],
    action_distribution: list[float] | None = None,
    update_rule_distribution: list[float] | None = None,
) -> dict[str, Any]:
    g = generator()
    wpdg = WeakPrisonerDilemmaGame(g, b, action_distribution, update_rule_distribution)
    results = {}
    for i in range(iterations):
        if i + 1 in retrieval_time:
            action, payoff, update_rule = wpdg.step(get_distributions=True)

            results[i + 1] = {
                "action": action,
                "payoff": sum(payoff),
                "update_rule": update_rule,
                "degree": list(dict(wpdg.g.degree()).values()),
            }
        else:
            wpdg.step()
    return results


def run_full_ultimatum_game(
    generator: Callable[[], nx.Graph],
    iterations_per_network: int,
    iterations_per_game_strategy: int,
    retrieval_time: list[int],
    update_rule_distribution: list[float] | None = None,
) -> dict[UGStrategy, list[ResultsDict]]:
    results = {}
    for game_strategy in UGStrategy:
        results[game_strategy] = Parallel(n_jobs=-1, verbose=10)(
            delayed(
                lambda: run_single_ultimatum_game(
                    generator,
                    game_strategy,
                    iterations_per_network,
                    retrieval_time,
                    update_rule_distribution,
                )
            )()
            for _ in range(iterations_per_game_strategy)
        )
    return results


def run_full_wpd_game(
    generator: Callable[[Any], nx.Graph],
    b_values: list[float],
    iterations_per_network: int,
    iterations_per_game_strategy: int,
    retrieval_time: list[int],
    action_distribution: list[float] | None = None,
    update_rule_distribution: list[float] | None = None,
):
    results = {}
    for b in b_values:
        results[b] = Parallel(n_jobs=-1, verbose=10)(
            delayed(
                lambda: run_single_wpd_game(
                    generator,
                    b,
                    iterations_per_network,
                    retrieval_time,
                    action_distribution,
                    update_rule_distribution,
                )
            )()
            for _ in range(iterations_per_game_strategy)
        )
    return results


def main():
    net_type = "ba"
    N = 1000
    k = 4
    l = 8
    p_in = 0.025
    p_out = 0.2
    p = 0.01
    iterations_per_network = 10000
    iterations_per_game_strategy = 10
    retrieval_time = [10**n for n in range(10) if 10**n <= iterations_per_network]

    def generator():
        match net_type:
            case "ba":
                return generate_barabasi_albert(N)
            case "er":
                return generate_erdos_renyi(N, p)
            case "sbm":
                return generate_hierarchical_sbm(N // l, l, p_in, p_out)
            case "ws":
                return generate_watts_strogatz(N, k, p)

    DATA_PATH = Path("data") / net_type
    if not DATA_PATH.exists():
        DATA_PATH.mkdir()

    run_simulations = False
    if run_simulations:
        results = run_full_ultimatum_game(
            generator,
            iterations_per_network,
            iterations_per_game_strategy,
            retrieval_time,
        )

        save_ug_strat_distribution(results, DATA_PATH, 20)
        save_strategy_space(results, DATA_PATH)
        save_degree_of_selection(results, DATA_PATH)
        save_ug_update_rule_distribution(results, DATA_PATH)

    FIGURE_PATH = Path("figures") / net_type
    if not FIGURE_PATH.exists():
        FIGURE_PATH.mkdir()

    plot_strategy_distribution(
        DATA_PATH / "EMP_p_dist.csv", FIGURE_PATH / "EMP_p_dist.png", "p"
    )
    plot_strategy_distribution(
        DATA_PATH / "PRG_p_dist.csv", FIGURE_PATH / "PRG_p_dist.png", "p"
    )
    plot_strategy_distribution(
        DATA_PATH / "RND_p_dist.csv", FIGURE_PATH / "RND_p_dist.png", "p"
    )
    plot_strategy_distribution(
        DATA_PATH / "RND_q_dist.csv", FIGURE_PATH / "RND_q_dist.png", "q"
    )

    plot_strategy_space(
        DATA_PATH / "RND_strat_space.csv",
        FIGURE_PATH / "RND_strat_space.png",
        f"8 {net_type.upper()}-nets",
    )

    plot_strategy_frequency(
        DATA_PATH / "strat_freq.csv", FIGURE_PATH / "strat_freq.png"
    )

    plot_update_rule_distribution(
        DATA_PATH / "EMP_update_rule_dist.csv", FIGURE_PATH / "EMP_update_rule_dist.png"
    )
    plot_update_rule_distribution(
        DATA_PATH / "PRG_update_rule_dist.csv", FIGURE_PATH / "PRG_update_rule_dist.png"
    )
    plot_update_rule_distribution(
        DATA_PATH / "RND_update_rule_dist.csv", FIGURE_PATH / "RND_update_rule_dist.png"
    )
    return


if __name__ == "__main__":
    main()
