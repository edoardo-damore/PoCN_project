import marimo

__generated_with = "0.19.7"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    import random

    from joblib import Parallel, delayed
    from pprint import pprint
    from typing import Literal, Any, Callable

    from analysis import plot_strategy_distribution, plot_strategy_space
    from games import WeakPrisonerDilemmaGame, UltimatumGame
    from generators import (
        generate_barabasi_albert,
        generate_erdos_renyi,
        generate_hierarchical_sbm,
        generate_watts_strogatz,
    )

    plt.style.use("default")
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 18
    })
    return (
        Any,
        Callable,
        Literal,
        Parallel,
        UltimatumGame,
        delayed,
        generate_barabasi_albert,
        plot_strategy_space,
    )


@app.cell
def _(Any, Callable, Literal, UltimatumGame):
    def run_ultimatum_game(
        generator: Callable,
        game_strategy: Literal["EMP", "PRG", "RND"],
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
                    "p": p,
                    "q": q,
                    "payoff": sum(payoff),
                    "update_rule": update_rule,
                    "degree": list(dict(ug.g.degree()).values()),
                }
            else:
                ug.step()
        return results
    return (run_ultimatum_game,)


@app.cell
def _():
    N = 1000
    k = 4
    l = 8
    p_in = 0.5
    p_out = 0.05
    p = 0.1
    iterations = 10000
    retrieval_time = [10**n for n in range(10) if 10**n <= iterations]
    return N, iterations, retrieval_time


@app.cell
def _(
    N,
    Parallel,
    delayed,
    generate_barabasi_albert,
    iterations,
    retrieval_time,
    run_ultimatum_game,
):
    results = Parallel(n_jobs=-1)(
        delayed(
            lambda: run_ultimatum_game(
                lambda: generate_barabasi_albert(N),
                "RND",
                iterations,
                retrieval_time,
            )
        )()
        for _ in range(12)
    )
    return (results,)


@app.cell
def _(plot_strategy_space, results):
    plot_strategy_space([result for result in results], 8)
    return


if __name__ == "__main__":
    app.run()
