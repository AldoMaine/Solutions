# -*- coding: utf-8 -*-
#  pylint: disable=
"""
Closest Insertion Algorithm

"""
from timeit import default_timer as timer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Vertex:
    """Vertex class."""

    def __init__(self, x, y, name, index, dists):
        self.x = x  # longitude
        self.y = y  # latitude
        self.name = name  # string name
        self.index = index  # reference index
        self.dists = dists  # link to a related adjacency matrix

    def distance(self, city):
        """Distance from this location to another."""
        distance = self.dists[self.index, city.index]
        return distance

    def __repr__(self):
        """."""
        return "(" + self.name + " " + str(self.x) + "," + str(self.y) + ")"


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


def shift_route(route, home):
    """Shift sequence of cities so that the home city is first.

    This makes it easier to compare solutions with other algorithms.
    (i.e. if the home city ends up in 5th place, move the first 4 cities
    to the end of the sequence to create a "shifted" route.

    Parameters
    ----------
    route : list
        Ordered list of vertex indices.
    home : int
        index of home city vertex

    Returns
    -------
    shifted : list
        list of Vertex object shifted so that Vertex 0 (home) is first.

    """
    # if route is already a full cycle, remove last city
    if route[-1] == route[0]:
        route = route[:-1]

    origin = 0
    for pos, val in enumerate(route):
        if val == home:
            origin = pos
            break
    shifted = route[origin:] + route[:origin]

    return shifted


def longest_leg(A, cycle):
    """
    Find longest leg in a cycle.

    Parameters
    ----------
    A : array
        distance array.  a_i,j is the distance from i to j.
    cycle : list
        ordered list of cycle verticies from start to finish.

    Returns
    -------
    longest : float
        longest edge in cycle.

    """
    if cycle[-1] != cycle[0]:  # not a completed cycle
        cycle.append(cycle[0])

    n = len(cycle)
    longest = 0
    for j in range(n-1):
        city1 = cycle[j]
        city2 = cycle[j + 1]
        if A[city1, city2] > longest:
            longest = A[city1, city2]

    return longest


def loc_plot(route, title):
    """Creates a map of city locations (as indices) without a route."""
    plt.figure()
    waypoints = np.empty((0, 2))
    for city in route:
        waypoints = np.append(waypoints, np.array([[city.x, city.y]]), axis=0)

    # add starting city to end of route to close the loop
    waypoints = np.append(waypoints, np.array([[route[0].x, route[0].y]]),
                          axis=0)

    try:
        for city in route:
            plt.annotate(city.index,
                         (city.x, city.y),  # coordinates to position the label
                         textcoords="offset points",  # how to position text
                         xytext=(0, 6),  # distance from text to points (x,y)
                         ha='center')  # horizontal alignment
    except AttributeError:
        pass

    plt.plot(waypoints[:, 0], waypoints[:, 1], 'rs')
    plt.title(title)


def route_plot(route, title, distance, cycles):
    """
    Plot a route (Hamiltonian cycle).

    Parameters
    ----------
    route : list
        Ordered list of vertex objects with x, y properties.
    title : string
        Plot title.
    distance : float
        Overall cycle distance for the route.
    cycles : int
        Number of loops or iterations needed to find route.

    """
    plt.figure()
    waypoints = np.empty((0, 2))
    for vertex in route:
        waypoints = np.append(waypoints, np.array([[vertex.x, vertex.y]]),
                              axis=0)

    # add starting vertex to end of route to close the loop
    waypoints = np.append(waypoints, np.array([[route[0].x, route[0].y]]),
                          axis=0)

    try:
        for vertex in route:  # x,y in zip(xs,ys):
            plt.annotate(vertex.name,
                         (vertex.x, vertex.y),  # label base coordinates
                         textcoords="offset points",  # how to position text
                         xytext=(0, 6),  # distance from text to points (x,y)
                         ha='center')  # horizontal alignment
    except AttributeError:
        pass

    plt.plot(waypoints[:, 0], waypoints[:, 1], 'rs-')
    plt.title(f"{title}\nDistance: {distance:.1f}, Iterations: {cycles}")


def closest_insertion_tsp(V, start, dists):
    """
    Closest Insertion Algorithm from "THE TRAVELING SALESMAN PROBLEM",
     Thesis by Corinne Brucato, University of Pittsburgh, 2013

    Similar to the nearest neighbor algorithm except it searches for a
    nearest neighbor from *both* ends of the current path.
    Results are dependent on starting vertex.

    G : a weighted, complete graph
    V : set of vertices in G
    P : path, an ordered list of vertices e.g. [A, D...]
    T : edges connected to ends of path, e.g. [AB, AC, AD...]

    Initialize:
        1. Pick a vertex u_i in V
        2. Form a path P = [u_i] and set T = [edges connected to u_i]

    Iterate:
        1a. Find edges in G incident with the beginning and end of the path,
            that are not already in T.
        1b. Choose the edge from step 1a that has the minimum weight (u_i, u_j)
            (Ties may be broken arbitrarily.).
        2a. Add the end vertex (u_j) from this edge to P.
        2b. Remove all edges connected to u_j from T.
        3. Repeat until T includes all the vertices in G

    Parameters
    ----------
    V : list
        list of integers representing N colleges, 0..N-1.
    start : int
        starting index for route search
    dists : array
        adjacency matrix with inter-vertex distances.

    Returns
    -------
    d : float
        Total cycle distance.
    route : list
        Ordered list of vertices.
    nsteps : int
        Number of iterations or loops required to find route.

    """
    # intialize
    ui = V[start]  # first vertex u_i from V
    #  Form a path P = {u_i} and set T = {edges connected to u_i}
    P = [ui]  # P must be a list or array to maintain order

    T = np.empty((0, 3))
    for index, dist in enumerate(dists[ui]):
        if ui != index and dist > 0:
            T = np.append(T, [[ui, index, dist]], axis=0)

    r = np.argmin(T[:, 2])
    j = int(T[r, 1])  # the last vertex added to P
    P.append(j)
    d = T[r, 2]
    T = np.delete(T, (r), axis=0)  # remove the edge just added to path from T

    nsteps = len(V)

    while True:
        # Find edges in G incident with the beginning and end of the path,
        #    that are not already in T.
        # j is the last vertex we added
        for index, dist in enumerate(dists[j]):
            if index not in P and dist > 0:
                T = np.append(T, [[j, index, dist]], axis=0)

        # Choose the edge from step 1a that has the minimum weight
        #     (Ties may be broken arbitrarily.)
        r = np.argmin(T[:, 2])
        # Put the end vertex from this edge (u_j) in P
        i = int(T[r, 0])
        j = int(T[r, 1])
        if i == P[0]:
            P.insert(0, j)
        else:
            P.append(j)

        d += T[r, 2]
        # remove all edges from T connected to i
        T = T[T[:, 0] != i]
        T = T[T[:, 1] != i]
        T = T[T[:, 1] != j]

        if len(P) == len(V):
            break

    d += dists[P[-1], P[0]]
    route = shift_route(P, start)
    route.append(start)

    return (d, route, nsteps)


def main():
    """Demonstrate closest insertion algorithm."""

    dists = pd.read_csv('colleges.txt', sep=r"\s", index_col=0,
                        engine='python').to_numpy()

    # create colleges collection for list of colleges and associated towns
    df = pd.read_csv("college_locs.csv", index_col=0)

    colleges = []
    num_cities = len(df.index)
    for i in range(0, num_cities):
        colleges.append(Vertex(df.iloc[i].long, df.iloc[i].lat,
                               df.iloc[i].college, i, dists))
    # loc_plot(colleges, "Colleges")  # for devel

    # Search for best starting city
    t0 = timer()
    options = np.empty((0, 2))
    for k in range(20):
        ans = closest_insertion_tsp(list(range(num_cities)), k, dists)
        options = np.append(options, [[k, ans[0]]], axis=0)

    tn = timer()

    plt.figure()
    plt.bar([str(x) for x in range(num_cities)], options[:, 1])
    plt.ylim(9500, 12500)
    plt.title('Cycle distance vs. Starting City')

    origin = np.argmin(options[:, 1])

    ans = closest_insertion_tsp(list(range(num_cities)), origin, dists)

    route = [colleges[i] for i in ans[1]]  # route as list of colleges
    print(f'\nProcessing time: {(tn-t0)/1e-3:.3g} msec for {ans[2]}'
          ' iterations.')
    print("Starting at ", colleges[origin].name)
    print('Path: ', ans[1])
    # print('Longest leg: ', longest_leg(dists, ans[1]))
    print('Closest Insertion distance: ', ans[0])

    route_plot(route, "Closest Insertion Solution", ans[0], ans[2])


if __name__ == "__main__":
    main()
