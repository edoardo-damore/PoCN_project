from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl


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
        lazy_df = pl.scan_csv(path, encoding="utf8-lossy", infer_schema_length=10000)
        return self._sanitize(lazy_df, config)

    def _sanitize(
        self, lazy_df: pl.LazyFrame, config: list[pl.Expr] | None
    ) -> pl.LazyFrame:
        """ """
        lazy_df = lazy_df.with_columns(config)
        return lazy_df


def remap_id(
    nodes: pl.LazyFrame,
    edges: pl.LazyFrame,
    old_id_name: str,
    new_id_name: str,
) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    if old_id_name not in nodes.collect_schema().keys():
        raise ValueError(
            f'old_id_name = "{old_id_name}" is not an existing column name'
        )
    if old_id_name == new_id_name:
        raise ValueError("old_id_name and new_id_name must have different values")

    mapping = (
        nodes.select(pl.col(old_id_name))
        .unique()
        .with_columns(pl.col(old_id_name).rank("ordinal").alias(new_id_name))
    )

    nodes = nodes.join(mapping, on=old_id_name).drop(pl.col(old_id_name))

    edges = (
        edges.join(
            mapping,
            left_on=f"ori_{old_id_name}",
            right_on=old_id_name,
        )
        .drop(pl.col(f"ori_{old_id_name}"))
        .rename({new_id_name: f"ori_{new_id_name}"})
        .join(
            mapping,
            left_on=f"des_{old_id_name}",
            right_on=old_id_name,
        )
        .drop(pl.col(f"des_{old_id_name}"))
        .rename({new_id_name: f"des_{new_id_name}"})
    )
    return nodes, edges


def divide_by_city(
    nodes: pl.LazyFrame, edges: pl.LazyFrame, cities: pl.Series, path: Path
) -> None:
    print("Generating nodes.csv and edges.csv for:")
    for city in cities:
        print(city)
        local_nodes = nodes.filter(pl.col("city") == city)
        local_edges = edges.join(
            local_nodes.select(pl.col("node")),
            left_on="ori_node",
            right_on="node",
        ).join(
            local_nodes.select(pl.col("node")),
            left_on="des_node",
            right_on="node",
        )
        local_nodes, local_edges = remap_id(local_nodes, local_edges, "node", "nodeid")
        local_nodes = local_nodes.select(
            pl.col("nodeid"), pl.col("lat"), pl.col("lon"), pl.col("layer")
        )
        local_edges = local_edges.select(
            pl.col("ori_nodeid"),
            pl.col("des_nodeid"),
            pl.col("ori_layer"),
            pl.col("des_layer"),
            pl.col("minutes"),
            pl.col("km"),
        )
        city_dir = path / city
        if not city_dir.exists() or not city_dir.is_dir():
            city_dir.mkdir()
        local_nodes.collect().write_csv(city_dir / "nodes.csv")
        local_edges.collect().write_csv(city_dir / "edges.csv")
    return


def main() -> None:
    BASE_DIR = Path(__file__).parent.parent.parent
    RAW_DIR = BASE_DIR / "raw" / "task_42"
    DATA_DIR = BASE_DIR / "data" / "task_42"
    if not DATA_DIR.exists() or not DATA_DIR.is_dir():
        DATA_DIR.mkdir()

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

    cities = raw_data.cities.filter(pl.col("pop2025") > 50000)
    nodes = pl.concat(
        [
            raw_data.nodes.join(raw_data.stops, left_on="atcocode", right_on="ATCOCode")
            .join(cities, left_on=f"{locality_type}Locality", right_on="city")
            .rename({f"{locality_type}Locality": "city"})
            .select(["node", "layer", "lat", "lon", "atcocode", "city"])
            for locality_type in ["GrandParent", "Parent", "NatGaz"]
        ]
    )

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

    cities = cities.select(pl.col("city")).collect().to_series()

    # given the high number of cities, we first collect the LazyFrames
    # this ensures that all transformations until now are done only one time
    # then we re-lazy them
    divide_by_city(nodes.collect().lazy(), edges.collect().lazy(), cities, DATA_DIR)


if __name__ == "__main__":
    main()
