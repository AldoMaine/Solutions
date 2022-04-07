# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 09:20:31 2022
https://sandipanweb.wordpress.com/2020/12/08/travelling-salesman-problem-tsp-with-python/
and corrected/edited by me
@author: aldon
"""
from timeit import default_timer as timer
from itertools import product
import pandas as pd
from mip import Model, xsum, minimize, BINARY
from tsp_evolutionary import College, route_plot


def TSP_ILP(G):
    """
    Travelling Salesman Problem solution using integer linear program.
    

    Parameters
    ----------
    G : array
        Graph data as an adjacency matrix.

    Returns
    -------
    TYPE
        DESCRIPTION.
    cycle : TYPE
        DESCRIPTION.

    """
    V1 = range(len(G))
    n, V = len(G), set(V1)
    model = Model()   # binary variables indicating if arc (i,j) is used
    # on the route or not
    x = [[model.add_var(var_type=BINARY) for j in V] for i in V]
    # continuous variable to prevent subtours: each city will have a
    # different sequential id in the planned route except the 1st one
    y = [model.add_var() for i in V]
    # objective function: minimize the distance
    model.objective = minimize(xsum(G[i][j]*x[i][j] for i in V for j in V))

    # constraint : leave each city only once
    for i in V:
        model += xsum(x[i][j] for j in V - {i}) == 1
        # constraint : enter each city only once
    for i in V:
        model += xsum(x[j][i] for j in V - {i}) == 1   # subtour elimination
    for (i, j) in product(V - {0}, V - {0}):
        if i != j:
            model += y[i] - (n+1)*x[i][j] >= y[j]-n   # optimizing

    model.verbose = 0
    model.optimize()   # checking if a solution was found

    if model.num_solutions:
        nc = 0  # cycle starts from vertex 0
        cycle = [nc]
        while True:
            nc = [i for i in V if x[nc][i].x >= 0.99][0]
            cycle.append(nc)
            if nc == 0:
                break

    return (model.objective_value, cycle)


def main():
    df_distances = pd.read_csv('colleges.txt', sep=r"\s", index_col=0,
                               engine='python')
    dists = df_distances.to_numpy()  # adjacency matrix
    
    # create colleges collection for list of colleges and associated towns
    df = pd.read_csv("college_locs.csv", index_col=0)
    colleges = []
    num_cities = len(df.index)
    for i in range(0, num_cities):
        colleges.append(College(df.iloc[i].long, df.iloc[i].lat,
                                df.iloc[i].college, i, dists))

    t0 = timer()
    ans = TSP_ILP(dists)
    tn = timer()
    
    print('Processing time: {:.3g} sec'.format((tn-t0)))
    print(f'Total distance {ans[0]}')
    print(f'Route {ans[1]}')

    route = [colleges[i] for i in ans[1]]  # route as list of colleges
    route_plot(route, "Integer Programming Solution", ans[0], "")

if __name__ == "__main__":
    main()
