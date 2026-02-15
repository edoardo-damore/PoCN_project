import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from joblib import Parallel, delayed
from pathlib import Path
from typing import Any, Callable

from visualization import (
    plot_strategy_distribution,
    plot_strategy_space,
    plot_strategy_frequency,
    plot_update_rule_distribution,
    plot_avg_strat_over_degree,
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
    generate_real_world,
)
from utils import (
    save_ug_strat_distribution,
    save_strategy_frequency,
    save_strategy_space,
    save_ug_update_rule_distribution,
    save_average_strategy_over_degree,
)

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 18,
        "text.latex.preamble": r"""
        \usepackage{amsmath}
        \usepackage{braket}
""",
    }
)


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
    # run options
    net_type = "rw" # choose between: ba, er, sbm, ws, rw
    run_simulations = False # False if you only want to plot, given that the simulation already ran
    
    # run parameters
    N = 5000
    k = 8
    l = 8
    p_in = 0.1
    p_out = 0.035
    p = 0.01
    iterations_per_network = 10000
    iterations_per_game_strategy = 100
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
            case "rw":
                return generate_real_world(Path("inf-power") / "inf-power.mtx")

    DATA_PATH = Path("data")
    if not DATA_PATH.exists():
        DATA_PATH.mkdir()
    DATA_PATH = Path("data") / net_type
    if not DATA_PATH.exists():
        DATA_PATH.mkdir()

    if run_simulations:
        results = run_full_ultimatum_game(
            generator,
            iterations_per_network,
            iterations_per_game_strategy,
            retrieval_time,
        )

        save_ug_strat_distribution(results, DATA_PATH, 20)
        save_strategy_space(results, DATA_PATH)
        save_strategy_frequency(results, DATA_PATH)
        save_ug_update_rule_distribution(results, DATA_PATH)
        if net_type == "ba" or net_type == "rw":
            save_average_strategy_over_degree(results, DATA_PATH)

    FIGURE_PATH = Path("figures")
    if not FIGURE_PATH.exists():
        FIGURE_PATH.mkdir()
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

    if net_type == "ba" or net_type == "rw":
        plot_avg_strat_over_degree(
            DATA_PATH / "EMP_avg_strat_over_degree.csv",
            FIGURE_PATH / "EMP_avg_strat_over_degree.png",
        )
        plot_avg_strat_over_degree(
            DATA_PATH / "PRG_avg_strat_over_degree.csv",
            FIGURE_PATH / "PRG_avg_strat_over_degree.png",
        )
        plot_avg_strat_over_degree(
            DATA_PATH / "RND_avg_strat_over_degree.csv",
            FIGURE_PATH / "RND_avg_strat_over_degree.png",
        )
    return


if __name__ == "__main__":
    main()
