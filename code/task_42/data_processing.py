from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from sklearn.neighbors import radius_neighbors_graph, BallTree
from scipy.sparse.csgraph import connected_components


class RawDataStorer:
    """
    Class acting as a container of all the necessary tables.
    Basic sanitization is also applied.
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
        lazy_df = lazy_df.with_columns(config)
        return lazy_df


def remap_id(
    nodes: pl.LazyFrame,
    edges: pl.LazyFrame,
    old_id_name: str,
    new_id_name: str,
) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """
    Remaps the old node ids into new ones that go from 1 to N
    """
    if old_id_name not in nodes.collect_schema().keys():
        raise ValueError(
            f'old_id_name = "{old_id_name}" is not an existing column name'
        )
    if old_id_name == new_id_name:
        raise ValueError("old_id_name and new_id_name must have different values")

    mapping = (
        nodes.select(pl.col(old_id_name)).unique().with_row_index(new_id_name, offset=1)
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
    """
    A new set of tables (nodes and edges) is generated for each city.
    """
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

    # defining the path for each file
    # IMPORTANT: while all other files should be in the same relative positions as they are originally collected,
    # the city population one is collected separately, so it must be added manually.
    # change RAW_DIR path accordingly
    raw_data_paths = {
        "nodes": RAW_DIR / "nodes.csv",
        "stops": RAW_DIR / "NaPTAN_NPTG" / "Stops.csv",
        "cities": RAW_DIR / "united-kingdom-cities-by-population-2025.csv",
        "groups": RAW_DIR
        / "NaPTAN_NPTG"
        / "Groups.csv",  # this is not actually used, just a relic of ancient experimentation with the dataset
        "edges": RAW_DIR / "edges.csv",
    }
    # defining basic attribute preprocessing
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

    # collecting the raw data from the csv files and applying the preprocessing
    raw_data = RawDataStorer(raw_data_paths, raw_data_configs)

    # selecting the cities with at least 50k inhabitants
    cities = raw_data.cities.filter(pl.col("pop2025") > 50000)

    # nodes that can be associated to a specific city given their locality attributes
    recognized_nodes = pl.concat(
        [
            raw_data.nodes.join(raw_data.stops, left_on="atcocode", right_on="ATCOCode")
            .join(cities, left_on=f"{locality_type}Locality", right_on="city")
            .rename({f"{locality_type}Locality": "city"})
            .select(["node", "layer", "lat", "lon", "atcocode", "city"])
            for locality_type in ["GrandParent", "Parent", "NatGaz"]
        ]
    )

    # adding the city attribute to the whole node set
    nodes = raw_data.nodes.join(
        recognized_nodes.select(["node", "city"]),
        left_on="node",
        right_on="node",
        how="left",
    )

    ### ***CODE DEVELOPED WITH THE HELP OF GEMINI 3 PRO***
    # In this section i used ai tools in oder to speed up the development due to time constraints
    #
    # The idea is to use the already labelled nodes to infer the city of their neighbors within a certain radius

    coords = np.deg2rad(nodes.select(["lat", "lon"]).collect().to_numpy())

    # create a radius neighbors graph
    radius_km = 1
    radius_rad = radius_km / 6371.0  # approximation for small angles
    adjacency = radius_neighbors_graph(
        coords, radius=radius_rad, metric="haversine", include_self=True
    )
    _, labels = connected_components(adjacency)

    # add the group attribute
    df_groups = nodes.with_columns(pl.Series("group_id", labels)).collect()

    # counting how many cities are in each group
    group_stats = df_groups.group_by("group_id").agg(
        [
            pl.col("city").drop_nulls().unique().alias("found_cities"),
            pl.col("city").drop_nulls().n_unique().alias("n_cities"),
        ]
    )

    # we split groups based on how many cities there are.
    # we ignore the case of 0 cities
    safe_groups = group_stats.filter(pl.col("n_cities") == 1)
    conflict_groups = group_stats.filter(pl.col("n_cities") > 1)

    # in the case of 1 city, then all nodes in the group are also associated to that city
    safe_mapping = safe_groups.select(
        ["group_id", pl.col("found_cities").list.first().alias("safe_city")]
    )

    df_safe = (
        df_groups.join(safe_mapping, on="group_id", how="inner")
        .with_columns(pl.coalesce(["city", "safe_city"]).alias("city"))
        .drop("safe_city")
    )

    # for more that 1 city, we assign each node to the closest city
    df_conflict = df_groups.filter(
        pl.col("group_id").is_in(conflict_groups["group_id"])
    )

    if df_conflict.height > 0:
        # 1. Separate the "Seeds" (Known) and "Targets" (Unknown) *within these groups*
        seeds = df_conflict.filter(pl.col("city").is_not_null())
        targets = df_conflict.filter(pl.col("city").is_null())

        if targets.height > 0:
            # 2. Build a local Tree on just the conflicting seeds
            seed_rads = np.deg2rad(seeds.select(["lat", "lon"]).to_numpy())
            target_rads = np.deg2rad(targets.select(["lat", "lon"]).to_numpy())

            tree = BallTree(seed_rads, metric="haversine")

            # 3. Find closest seed for each target
            _, indices = tree.query(target_rads, k=1)

            # 4. Map indices back to city names
            resolved_cities = seeds[indices.flatten()]["city"]

            # 5. Assign
            targets = targets.with_columns(resolved_cities.alias("city"))

            # 6. Recombine seeds and filled targets
            df_conflict_solved = pl.concat([seeds, targets])
        else:
            df_conflict_solved = seeds
    else:
        df_conflict_solved = df_conflict  # Empty

    nodes = (
        pl.concat([df_safe, df_conflict_solved]).sort("node").drop("group_id")
    ).lazy()  # re-lazy for consistency

    ### ***END OF AI GENERATED CODE***

    # given the nodes found, we select all edges that connect them
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

    # transforming the LazyFrame into Series
    cities = cities.select(pl.col("city")).collect().to_series()

    # given the high number of cities, we first collect the LazyFrames
    # this ensures that all transformations until now are done only one time
    # then we re-lazy them
    divide_by_city(nodes, edges.collect().lazy(), cities, DATA_DIR)


if __name__ == "__main__":
    main()
