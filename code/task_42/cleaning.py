import marimo

__generated_with = "0.17.8"
app = marimo.App(
    width="full",
    css_file="/home/edoardo-damore/.local/share/motheme/themes/coldme.css",
)


@app.cell
def _():
    import marimo as mo

    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    import polars as pl

    from pathlib import Path
    from pydantic import validate_call
    return Path, mo, pl, plt


@app.cell
def _(Path):
    BASE_DIR = Path(__file__).parent.parent.parent
    RAW_DIR = BASE_DIR / "raw" / "task_42"
    DATA_DIR = BASE_DIR / "data" / "task_42"
    if not DATA_DIR.exists() or not DATA_DIR.is_dir():
        DATA_DIR.mkdir()
    return (RAW_DIR,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Retrieval and cleaning
    The first step is to retrieve the data and perform an immediate sanitization
    """)
    return


@app.cell
def _(Path, pl):
    class RawDataStorer:
        """
        Simple class to store and preprocess all the original data tables.

        # Parameters
        :param paths: the paths where the tables are stored. A LazyFrame is created for each table.
        :type paths: dict[str, Path]
        :param configs: lists of Polars' expressions that are applyed to the corresponding LazyFrames.
        :type configs: dict[str, list[pl.Expr]]
        """

        def __init__(
            self,
            paths: dict[str, Path],
            configs: dict[str, list[pl.Expr]],
        ):
            self.nodes = self._load(paths["nodes"], configs["nodes"])
            self.stops = self._load(paths["stops"], configs["stops"])
            self.cities = self._load(paths["cities"], configs["cities"])
            self.groups = self._load(paths["groups"], configs["groups"])
            self.edges = self._load(paths["edges"], configs["edges"])
            return

        def _load(self, path: Path, config: list[pl.Expr] | None) -> pl.LazyFrame:
            lazy_df = pl.scan_csv(
                path, encoding="utf8-lossy", infer_schema_length=10000
            )
            return self._sanitize(lazy_df, config)

        def _sanitize(
            self, lazy_df: pl.LazyFrame, config: list[pl.Expr] | None
        ) -> pl.LazyFrame:
            """ """
            lazy_df = lazy_df.with_columns(config)
            return lazy_df
    return (RawDataStorer,)


@app.cell
def _(RAW_DIR, RawDataStorer, pl):
    raw_data_paths = {
        "nodes": RAW_DIR / "nodes.csv",
        "stops": RAW_DIR / "NaPTAN_NPTG" / "Stops.csv",
        "cities": RAW_DIR / "united-kingdom-cities-by-population-2025.csv",
        "groups": RAW_DIR / "NaPTAN_NPTG" / "Groups.csv",
        "edges": RAW_DIR / "edges.csv",
    }
    raw_data_configs = {
        "nodes": [
            pl.col("node").str.strip_chars().cast(pl.Int32),
            pl.col("layer").cast(pl.Int32),
            pl.col("lon").str.strip_chars().cast(pl.Float64),
            pl.col("zone").str.strip_chars(),
            pl.col("atcocode").str.strip_chars(),
        ],
        "stops": [pl.col("ATCOCode").cast(pl.String)],
        "cities": None,
        "groups": None,
        "edges": [
            pl.col("ori_node").str.strip_chars().cast(pl.Int32),
            pl.col("des_node").str.strip_chars().cast(pl.Int32),
            pl.col("minutes").str.strip_chars().cast(pl.Int32),
            pl.col("km").str.strip_chars().cast(pl.Float64),
        ],
    }
    raw_data = RawDataStorer(raw_data_paths, raw_data_configs)
    raw_data.edges.collect().head(10)
    return (raw_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Creating the nodes
    """)
    return


@app.cell
def _(pl, raw_data):
    big_cities = raw_data.cities.filter(pl.col("pop2025") > 50000)
    all_nodes = raw_data.nodes.join(
        raw_data.stops, left_on="atcocode", right_on="ATCOCode"
    )
    nodes = pl.concat(
        [
            raw_data.nodes.join(
                raw_data.stops, left_on="atcocode", right_on="ATCOCode"
            )
            .join(big_cities, left_on=f"{locality_type}Locality", right_on="city")
            .rename({f"{locality_type}Locality": "city"})
            .select(["node", "layer", "lat", "lon", "atcocode", "city"])
            for locality_type in ["GrandParent", "Parent", "NatGaz"]
        ]
    )
    nodes.collect()
    return big_cities, nodes


@app.cell
def _(big_cities, nodes, plt):
    cities = big_cities.collect()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(nodes["lon"], nodes["lat"], marker=".", color="c", s=1)
    ax.scatter(cities["longitude"], cities["latitude"], marker=".", color="r", s=5)

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Selecting the edges
    """)
    return


@app.cell
def _(nodes, raw_data):
    edges = (
        raw_data.edges.join(
            nodes, left_on=["ori_node", "ori_layer"], right_on=["node", "layer"]
        )
        .rename({"city": "ori_city"})
        .join(nodes, left_on=["des_node", "des_layer"], right_on=["node", "layer"])
        .rename({"city": "des_city"})
        .select(
            [
                "ori_node",
                "des_node",
                "ori_layer",
                "des_layer",
                "ori_city",
                "des_city",
                "minutes",
                "km",
            ]
        )
    )
    edges.collect()
    return (edges,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Divide in cities
    """)
    return


@app.cell
def _(pl):
    def remap_key(
        data: pl.DataFrame,
        col: str,
        mapping: pl.DataFrame = None,
        return_mapping: bool = False,
    ) -> tuple[pl.DataFrame, pl.DataFrame] | pl.DataFrame:
        """
        Maps the specified column `col` into a new key

        Args:
            data (pl.DataFrame):
            col (str):
            mapping (pl.DataFrame, default = None): if `None` it will be generated automatically
            return_mapping (bool, default = False):
        Returns:
            pl.DataFrame: the mapped `data`
            pl.DataFrame (optional): the `mapping` (only if `return_mapping = True`)
        """
        if mapping is None:
            key_values = data[col].unique()
            mapping = pl.DataFrame(
                [key_values, pl.arange(1, key_values.shape[0] + 1, eager=True)],
                schema=[col, "newKey"],
            )
        mapped_data = data.join(mapping, on=col).drop(col).rename({"newKey": col})
        if return_mapping:
            return mapped_data, mapping
        return mapped_data
    return (remap_key,)


@app.cell
def _(data_path, os, pl, remap_key):
    def divide_by_city(data: dict[str, pl.DataFrame]) -> None:
        cities = (
            data["cities"]
            .filter(pl.col("pop2025") > 50000)
            .get_column("city")
            .to_list()
        )
        for city in cities:
            localNodes = (
                data["nodes"]
                .rename({"ParentLocality": "city"})
                .filter(pl.col("city") == city)
            )
            localEdges = data["edges"].filter(
                (pl.col("ori_city") == city) | (pl.col("des_city") == city)
            )
            if localNodes.shape[0] > 0:
                localNodes, mapping = remap_key(
                    localNodes, "node", return_mapping=True
                )
                localEdges = localEdges.pipe(
                    remap_key,
                    "ori_node",
                    mapping.rename({"node": "ori_node"}),
                    return_mapping=False,
                ).pipe(
                    remap_key,
                    "des_node",
                    mapping.rename({"node": "des_node"}),
                    return_mapping=False,
                )
            city_path = f"{data_path}/{city}"
            if not os.path.isdir(city_path):
                os.mkdir(city_path)
            localNodes.write_csv(f"{city_path}/nodes.csv")
            localEdges.write_csv(f"{city_path}/edges.csv")
        return
    return (divide_by_city,)


@app.cell
def _(divide_by_city, edges, nodes, raw_data_cities):
    divide_by_city({"cities": raw_data_cities, "nodes": nodes, "edges": edges})
    return


if __name__ == "__main__":
    app.run()
