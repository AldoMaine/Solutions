# -*- coding: utf-8 -*-
"""
The Nearest Neighbor traveling salesman algorithm builds a route by always
choosing the nearest unvisited vertex. It is very fast but rarely optimal.
Results are affected by starting vertex.

"""
from timeit import default_timer as timer
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tsp_evolutionary import College, route_plot


def nearest_neighbor_cycle(A, origin):
    """Nearest neighbor algorithm for travelling salesman problem (TSP).

    A complete graph (every vertex is commented to every other vertex)
    is assumed.

    Modified code from answer provided by Luke in:
    https://stackoverflow.com/questions/17493494/nearest-neighbour-algorithm

    Parameters
    ----------
    A : array
        an NxN array indicating distance between N vertices.
    origin : int
        index of the starting vertex.

    Returns
    -------
    route : list
        ordered list of vertices defining a solution to the TSP.
    cost_nn : float
        Cost (travel distance) of solution.

    """

    route = [origin]  # initialize route and cost
    cost_nn = 0
    N = A.shape[0]  # N cities to visit
    mask = np.ones(N, dtype=bool)  # boolean values indicating which
    # locations have not been visited
    mask[origin] = False

    for _i in range(N-1):
        last = route[-1]  # last city visited
        next_ind = np.argmin(A[last][mask])  # find minimum of remaining locs
        next_loc = np.arange(N)[mask][next_ind]  # convert to original location
        route.append(next_loc)
        mask[next_loc] = False
        cost_nn += A[last, next_loc]

    route.append(origin)  # return to origin
    cost_nn += A[next_loc, origin]

    return route, cost_nn


def main():
    """Demonstrate nearest neighbor algorithm."""

    dists = pd.read_csv('colleges.txt', sep=r"\s", index_col=0,
                        engine='python').to_numpy()

    # create colleges collection for list of colleges and associated towns
    df = pd.read_csv("college_locs.csv", index_col=0)

    colleges = []
    num_cities = len(df.index)
    for i in range(0, num_cities):
        colleges.append(College(df.iloc[i].long, df.iloc[i].lat,
                                df.iloc[i].college, i, dists))
    # loc_plot(colleges, "Colleges")

    # Search for best starting city
    t0 = timer()
    options = np.empty((0, 2))
    for k in range(num_cities):
        ans = nearest_neighbor_cycle(dists, k)
        options = np.append(options, [[k, ans[1]]], axis=0)
        # print(f'start: {k}, distance: {ans[1]}')
    tn = timer()
    """Since starting city affect result, time should include a loop for
    each possible starting city. """

    plt.figure()
    plt.bar([str(x) for x in range(num_cities)], options[:, 1])
    plt.ylim(10000, 12500)
    plt.title('Route distance vs. Starting City')
    plt.xlabel('Starting Point (Index)')
    plt.ylabel('Route Distance (mi)')

    origin = np.argmin(options[:, 1])
    ans = nearest_neighbor_cycle(dists, origin)

    route = [colleges[i] for i in ans[0]]  # route as list of colleges
    print('\nProcessing time: {:.3g} msec for {} iterations.'.format(
        (tn-t0)/1e-3, num_cities-1))
    print("Starting at ", colleges[origin].name)
    print('Path: ', ans[0])
    # print('Longest leg: ', longest_leg(dists, ans[1]))
    print('Nearest Neighbor distance: ', ans[1])

    route_plot(route, "Nearest Neighbor Solution", ans[1], num_cities-1)


if __name__ == "__main__":
    main()
