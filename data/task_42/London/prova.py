import marimo

__generated_with = "0.19.7"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    import polars as pl
    return mo, pl


@app.cell
def _(pl):
    nodes = pl.read_csv("nodes.csv")
    edges = pl.read_csv("edges.csv")
    return (edges,)


@app.cell
def _(edges, mo):
    _df = mo.sql(
        f"""
        DESCRIBE edges
        """
    )
    return


if __name__ == "__main__":
    app.run()
