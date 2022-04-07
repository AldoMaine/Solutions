# -*- coding: utf-8 -*-
"""
http://matejgazda.com/christofides-algorithm-in-python/
https://github.com/BraveDistribution/pytsp/tree/master/pytsp

additions by Al Duke
"""

import itertools

import numpy as np
import pandas as pd
import networkx as nx
from timeit import default_timer as timer
from tsp_evolutionary import College, route_plot

from networkx.algorithms.matching import max_weight_matching
from networkx.algorithms.euler import eulerian_circuit

from utils import minimal_spanning_tree


def christofides_tsp(graph, starting_node=0):
    """
    Christofides TSP algorithm
    http://www.dtic.mil/dtic/tr/fulltext/u2/a025602.pdf
    Args:
        graph: 2d numpy array matrix
        starting_node: of the TSP
    Returns:
        tour given by christofies TSP algorithm

    Examples:
        >>> import numpy as np
        >>> graph = np.array([[  0, 300, 250, 190, 230],
        >>>                   [300,   0, 230, 330, 150],
        >>>                   [250, 230,   0, 240, 120],
        >>>                   [190, 330, 240,   0, 220],
        >>>                   [230, 150, 120, 220,   0]])
        >>> christofides_tsp(graph)
    """

    mst = minimal_spanning_tree(graph, 'Prim', starting_node=0)
    odd_degree_nodes = list(_get_odd_degree_vertices(mst))
    odd_degree_nodes_ix = np.ix_(odd_degree_nodes, odd_degree_nodes)
    nx_graph = nx.from_numpy_array(-1 * graph[odd_degree_nodes_ix])
    matching = max_weight_matching(nx_graph, maxcardinality=True)
    euler_multigraph = nx.MultiGraph(mst)
    for edge in matching:
        euler_multigraph.add_edge(odd_degree_nodes[edge[0]], odd_degree_nodes[edge[1]],
                                  weight=graph[odd_degree_nodes[edge[0]]][odd_degree_nodes[edge[1]]])
    euler_tour = list(eulerian_circuit(euler_multigraph, source=starting_node))
    path = list(itertools.chain.from_iterable(euler_tour))
    return _remove_repeated_vertices(path, starting_node)[:-1]


def _get_odd_degree_vertices(graph):
    """
    Finds all the odd degree vertices in graph
    Args:
        graph: 2d np array as adj. matrix

    Returns:
    Set of vertices that have odd degree
    """
    odd_degree_vertices = set()
    for index, row in enumerate(graph):
        if len(np.nonzero(row)[0]) % 2 != 0:
            odd_degree_vertices.add(index)
    return odd_degree_vertices


def _remove_repeated_vertices(path, starting_node):
    path = list(dict.fromkeys(path).keys())
    path.append(starting_node)
    return path


def get_distance(route, dists):
    """
    Compute overall distance for a route or cycle.

    Parameters
    ----------
    route : list
        Ordered list of vertex indices.
    dists : array
        Adjacency matrix. dists_i,j is the distance from i to j.

    Returns
    -------
    cost : float
        Total route cost.

    """
    cost = 0
    if route[0] != route[-1]:
        route.append(route[0])

    for i in range(len(route)-1):
        cost += dists[route[i], route[i+1]]
    # cost += dists[route[-1], route[0]]
    return cost

# graph = np.array([[  0, 300, 250, 190, 230],
#                   [300,   0, 230, 330, 150],
#                   [250, 230,   0, 240, 120],
#                   [190, 330, 240,   0, 220],
#                   [230, 150, 120, 220,   0]])
# ans = christofides_tsp(graph)


df_distances = pd.read_csv('colleges.txt', sep=r"\s", index_col=0,
                           engine='python')
dists = df_distances.to_numpy()  # adjacency matrix

t0 = timer()
route = christofides_tsp(dists)
tn = timer()

print('\nProcessing time: {:.3g} msec'.format((tn-t0)/1e-3))
print(route)
print(get_distance(route, dists))

"""runs fast but sub optimal results on a par with nearest neighbor"""






