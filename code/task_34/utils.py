import numpy as np
import polars as pl

from pathlib import Path

from custom_types import UGStrategy, ResultsDict, UpdateRule


def save_ug_strat_distribution(
    results: dict[UGStrategy, list[ResultsDict]],
    dir_path: Path,
    bins: int = 20,
):
    """
    SCHEMA:
    ```
    +---------+---+---+---+---+
    |bin_edges|t_1|t_2|...|t_N|
    +---------+---+---+---+---+
    ```
    """
    bin_edges = np.linspace(0, 1, bins + 1)
    retrieval_time = list(results[UGStrategy.EMP][0].keys())

    # saving the p distributions for every strat
    for game_strategy in UGStrategy:
        strat_results = results[game_strategy]
        counts = np.zeros(shape=(len(retrieval_time) + 1, bins + 1))
        counts[1:, 0] = np.array(retrieval_time)
        counts[0, :] = bin_edges
        for result in strat_results:
            for i, t in enumerate(retrieval_time):
                counts[i + 1, 1:] += np.histogram(
                    result[t]["strategy"][:, 0], bin_edges
                )[0]
        strat_df = pl.DataFrame(
            counts.T,
            schema=["bin_edges"] + [f"t_{i}" for i in range(len(retrieval_time))],
        )
        strat_df.write_csv(dir_path / f"{game_strategy.name}_p_dist.csv")

    # saving the q distribution for RND strat
    game_strategy = UGStrategy.RND

    strat_results = results[game_strategy]
    counts = np.zeros(shape=(len(retrieval_time) + 1, bins + 1))
    counts[1:, 0] = np.array(retrieval_time)
    counts[0, :] = bin_edges
    for result in strat_results:
        for i, t in enumerate(retrieval_time):
            counts[i + 1, 1:] += np.histogram(result[t]["strategy"][:, 1], bin_edges)[0]
    strat_df = pl.DataFrame(
        counts.T,
        schema=["bin_edges"] + [f"t_{i}" for i in range(len(retrieval_time))],
    )
    strat_df.write_csv(dir_path / f"{game_strategy.name}_q_dist.csv")

    return


def save_ug_update_rule_distribution(
    results: dict[UGStrategy, list[ResultsDict]],
    dir_path: Path,
):
    """
    SCHEMA:
    ```
    +-----------+---+---+---+---+
    |update_rule|t_1|t_2|...|t_N|
    +-----------+---+---+---+---+
    ```
    """
    retrieval_time = list(results[UGStrategy.EMP][0].keys())

    for game_strategy in UGStrategy:
        strat_results = results[game_strategy]
        counts = np.zeros(shape=(len(retrieval_time) + 1, len(UpdateRule) + 1))
        counts[1:, 0] = np.array(retrieval_time)
        counts[0, 1:] = np.array([update_rule.value for update_rule in UpdateRule])
        for result in strat_results:
            for i, t in enumerate(retrieval_time):
                for j, update_rule in enumerate(UpdateRule):
                    counts[i + 1, j + 1] += np.sum(
                        result[t]["update_rule"] == update_rule
                    )
        strat_df = pl.DataFrame(
            counts.T,
            schema=["update_rule"] + [f"t_{i}" for i in range(len(retrieval_time))],
        )
        strat_df.write_csv(dir_path / f"{game_strategy.name}_update_rule_dist.csv")

    return


def save_strategy_space(
    results: dict[UGStrategy, list[ResultsDict]],
    dir_path: Path,
    n_realizations: int = 8,
):
    trials = len(results[UGStrategy.EMP])
    retrieval_time = list(results[UGStrategy.EMP][0].keys())
    realizations = np.random.choice(
        np.arange(trials), size=n_realizations, replace=False
    )
    strat = np.concat(
        [
            results[UGStrategy.RND][i][retrieval_time[-1]]["strategy"]
            for i in realizations
        ],
        axis=0,
    )
    strat_df = pl.DataFrame(strat.T, schema=["p", "q"])
    strat_df.write_csv(dir_path / f"{UGStrategy.RND.name}_strat_space.csv")
    return


def save_degree_of_selection(
    results: dict[UGStrategy, list[ResultsDict]],
    dir_path: Path,
    max_strategies: int = 10,
):
    retrieval_time = list(results[UGStrategy.EMP][0].keys())
    t = retrieval_time[-1]
    N = results[UGStrategy.EMP][0][t]["strategy"].shape[0]

    unique_strat_counts = {}
    for game_strat in UGStrategy:
        unique_strat_counts[game_strat] = np.zeros((1, max_strategies))
        for result in results[game_strat]:
            _, counts = np.unique(result[t]["strategy"], axis=0, return_counts=True)
            counts = np.sort(counts)[::-1]
            if len(counts) < max_strategies:
                counts = np.pad(
                    counts,
                    (0, max_strategies - len(counts)),
                    constant_values=0,
                )
            else:
                counts = counts[:max_strategies]
            unique_strat_counts[game_strat] = np.concat(
                [unique_strat_counts[game_strat], counts[np.newaxis, :]], axis=0
            )
        unique_strat_counts[game_strat] = unique_strat_counts[game_strat][1:, :] / N

    df = pl.DataFrame(
        np.concat(
            [
                np.concat(
                    [
                        np.mean(unique_strat_counts[game_strat], axis=0).reshape(-1, 1),
                        np.std(unique_strat_counts[game_strat], axis=0).reshape(-1, 1),
                    ],
                    axis=1,
                )
                for game_strat in UGStrategy
            ],
            axis=1,
        ),
        schema=[
            f"{UGStrategy.EMP.name}_mean",
            f"{UGStrategy.EMP.name}_std",
            f"{UGStrategy.PRG.name}_mean",
            f"{UGStrategy.PRG.name}_std",
            f"{UGStrategy.RND.name}_mean",
            f"{UGStrategy.RND.name}_std",
        ],
    )
    df.write_csv(dir_path / "strat_freq.csv")
    return
