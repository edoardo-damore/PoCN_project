import networkx as nx

from generators import generate_real_world
from pathlib import Path

if __name__ == "__main__":
    g = generate_real_world(Path("inf-power") / "inf-power.mtx")
    print(g.degree())
