import networkx as nx
import numpy as np


def generate_hierarchical_sbm(
    n: int, l: int, p_in: float, p_out: float, max_iterations: int = 1000
) -> nx.Graph:
    if np.log2(l) % 1 != 0:
        l = int(2 ** np.floor(np.log2(l)))
        print(f"l will be rounded down to the nearest power of 2: {l}")
    sizes = [n] * l
    edge_prob = p_in * np.ones(shape=(l, l))
    div = 1
    while div < l:
        mult = p_out * np.ones(shape=(l, l))
        for i in range(l // div):
            mult[i * div : (i + 1) * div, i * div : (i + 1) * div] = 1
        edge_prob *= mult
        div *= 2
    g = nx.stochastic_block_model(sizes, edge_prob)
    iteration = 0
    while iteration < max_iterations and not nx.is_connected(g):
        g = nx.stochastic_block_model(sizes, edge_prob)
        iteration += 1
    return g


def generate_barabasi_albert(N: int) -> nx.Graph:
    return nx.barabasi_albert_graph(N, 1)


def generate_erdos_renyi(N: int, p: float) -> nx.Graph:
    return nx.erdos_renyi_graph(N, p)


def generate_watts_strogatz(N: int, k: int, p: float) -> nx.Graph:
    return nx.watts_strogatz_graph(N, k, p)
