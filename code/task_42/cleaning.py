import marimo

__generated_with = "0.16.5"
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
    import os
    import polars as pl
    return mo, os, pl, plt


@app.cell
def _(os):
    # this notebook has to be run from the project directory, not inside the `code` directory
    rawData_path = os.path.abspath("raw/task_42")
    data_path = os.path.abspath("data/task_42")
    if not os.path.isdir(data_path):
        os.makedirs(data_path)
    return data_path, rawData_path


@app.cell
def _(mo):
    mo.md(
        r"""
    # Retrieval and cleaning
    The first step is to retrieve the data and perform an immediate sanitization
    """
    )
    return


@app.cell
def _(pl):
    def sanitizeColumns(
        df: pl.DataFrame, sanitizeOptions: dict[str, list[str]] = None
    ) -> pl.DataFrame:
        """
        Sanitization pipeline:
        `remove_spaces` -> `to_int` -> `to_float`

        Every step is optional.
        Error handling is not yet supported, so be careful

        Args:
            df (pl.DataFrame): the DataFrame you would like to sanitize
            sanitizeOptions (dict[str, list[str]]): a dictionary where each key is a column of `df`, and the corresponding value is a list of sanitization steps (possible: `remove_spaces`, `to_int`, `to_float`)

        Returns:
            pl.DataFrame: the sanitized `df`
        """
        if sanitizeOptions is None:
            return df
        for col in sanitizeOptions.keys():
            if "remove_spaces" in sanitizeOptions[col]:
                df = df.with_columns(df.get_column(col).str.replace_all(" ", ""))
            if "to_int" in sanitizeOptions[col]:
                df = df.with_columns(df.get_column(col).cast(pl.Int32))
            if "to_float" in sanitizeOptions[col]:
                df = df.with_columns(df.get_column(col).cast(pl.Float64))
        return df
    return (sanitizeColumns,)


@app.cell
def _(pl, sanitizeColumns):
    def get_rawData(
        path: str, sanitizeOptions: dict[str, list[str]] = None, **kwargs
    ) -> pl.DataFrame:
        rawData = pl.read_csv(path, **kwargs).pipe(
            sanitizeColumns, sanitizeOptions
        )
        return rawData
    return (get_rawData,)


@app.cell
def _(get_rawData, rawData_path):
    rawData_nodes_sanitizeOptions = {
        "node": ["remove_spaces", "to_int"],
        "layer": ["to_int"],
        "lat": [],
        "lon": ["remove_spaces", "to_float"],
        "zone": ["remove_spaces"],
        "atcocode": ["remove_spaces"],
    }
    rawData_nodes = get_rawData(
        f"{rawData_path}/nodes.csv", rawData_nodes_sanitizeOptions
    )
    rawData_nodes
    return (rawData_nodes,)


@app.cell
def _(get_rawData, rawData_path):
    rawData_stops_sanitizeOptions = None
    rawData_stops = get_rawData(
        f"{rawData_path}/NaPTAN_NPTG/Stops.csv",
        rawData_stops_sanitizeOptions,
        encoding="utf8-lossy",
        schema_overrides={"ATCOCode": str},
    )
    rawData_stops
    return (rawData_stops,)


@app.cell
def _(get_rawData, rawData_path):
    rawData_cities_sanitizeOptions = None
    rawData_cities = get_rawData(
        f"{rawData_path}/united-kingdom-cities-by-population-2025.csv",
        rawData_cities_sanitizeOptions,
    )
    rawData_cities
    return (rawData_cities,)


@app.cell
def _(get_rawData, rawData_path):
    rawData_groups_sanitizeOptions = None
    rawData_groups = get_rawData(
        f"{rawData_path}/NaPTAN_NPTG/Groups.csv",
        rawData_groups_sanitizeOptions,
        encoding="utf8-lossy",
    )
    rawData_groups
    return (rawData_groups,)


@app.cell
def _(mo):
    mo.md(r"""# Creating the nodes""")
    return


@app.cell
def _(pl):
    def get_nodes(rawData: dict[str, pl.DataFrame]):
        stops = (
            rawData["nodes"]
            .join(rawData["stops"], left_on="atcocode", right_on="ATCOCode")
            .join(rawData["cities"], left_on="ParentLocality", right_on="city")
            .filter(pl.col("pop2025") > 50000)
            .select(["node", "layer", "lat", "lon", "atcocode", "ParentLocality"])
        )
        return stops
    return (get_nodes,)


@app.cell(hide_code=True)
def _():
    """
    def get_nodes(rawData: dict[str, pl.DataFrame]):
        # let's extract first the cities with more than 50k inhabitants
        # and let's create the corresponding regex pattern
        # AIDED WITH CLAUDE SONNET 4.5
        cities_list = (
            rawData["cities"]
            .filter(pl.col("pop2025") > 50000)
            .get_column("city")
            .to_list()
        )
        cities_pattern = "|".join(cities_list)

        # let's extract the stops present in the corresponding cities
        stops = (
            rawData["nodes"]
            .join(
                rawData["stops"].with_columns(
                    pl.concat_str(
                        [
                            # pl.col("CommonName"),
                            # pl.col("Identifier"),
                            pl.col("NatGazLocality"),
                            pl.col("ParentLocality"),
                            pl.col("GrandParentLocality"),
                        ],
                        separator=" - ",
                    ).alias("Locality")
                ),
                left_on="atcocode",
                right_on="ATCOCode",
            )
            .filter(pl.col("Locality").str.contains(cities_pattern))
            .select(["node", "layer", "lat", "lon", "atcocode", "Locality"])
        )

        # and now the groups
        groups = (
            rawData["nodes"]
            .join(rawData["groups"], left_on="atcocode", right_on="GroupID")
            .filter(pl.col("GroupName").str.contains(cities_pattern))
            .with_columns(pl.col("GroupName").alias("Locality"))
            .select(["node", "layer", "lat", "lon", "atcocode", "Locality"])
        )
        return pl.concat([stops, groups])
    """
    return


@app.cell
def _(get_nodes, rawData_cities, rawData_groups, rawData_nodes, rawData_stops):
    nodes = get_nodes(
        {
            "nodes": rawData_nodes,
            "stops": rawData_stops,
            "groups": rawData_groups,
            "cities": rawData_cities,
        }
    )
    return (nodes,)


@app.cell
def _(nodes, pl, plt, rawData_cities):
    cities = rawData_cities.filter(pl.col("pop2025") > 50000)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(nodes["lon"], nodes["lat"], marker=".", s=1)
    ax.scatter(cities["longitude"], cities["latitude"], marker=".", color="r", s=5)

    plt.show()
    return


@app.cell
def _(nodes):
    nodes
    return


@app.cell
def _(mo):
    mo.md(r"""# Selecting the edges""")
    return


@app.cell
def _(get_rawData, rawData_path):
    rawData_edges_sanitizeOptions = {
        "ori_node": ["remove_spaces", "to_int"],
        "des_node": ["remove_spaces", "to_int"],
        "minutes": ["remove_spaces", "to_int"],
        "km": ["remove_spaces", "to_float"],
    }
    rawData_edges = get_rawData(
        f"{rawData_path}/edges.csv", rawData_edges_sanitizeOptions
    )
    rawData_edges
    return (rawData_edges,)


@app.cell
def _(nodes, pl, rawData_edges):
    edges = (
        rawData_edges.join(
            nodes, left_on=["ori_node", "ori_layer"], right_on=["node", "layer"]
        )
        .with_columns(pl.col("ParentLocality").alias("ori_city"))
        .join(nodes, left_on=["des_node", "des_layer"], right_on=["node", "layer"])
        .with_columns(pl.col("ParentLocality").alias("des_city"))
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
    edges
    return (edges,)


@app.cell
def _(mo):
    mo.md(r"""# Divide in cities""")
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
        mapped_data = (
            data.join(mapping, on=col)
            .drop(col)
            .rename({"newKey": col})
        )
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
def _(divide_by_city, edges, nodes, rawData_cities):
    divide_by_city({"cities": rawData_cities, "nodes": nodes, "edges": edges})
    return


@app.cell
def _():
    print("a")
    return


if __name__ == "__main__":
    app.run()
