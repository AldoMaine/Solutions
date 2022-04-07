# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 13:21:19 2022

@author: aldon
"""

# traveling salesman
# problem using naive approach.
from sys import maxsize
from itertools import permutations
import pandas as pd

# implementation of traveling Salesman Problem
def travellingSalesmanProblem(graph, s):

    # store all vertex apart from source vertex
    vertex = []
    for i in range(V):
        if i != s:
            vertex.append(i)

    # store minimum weight Hamiltonian Cycle
    min_path = maxsize
    next_permutation=permutations(vertex)
    for i in next_permutation:

        # store current Path weight(cost)
        current_pathweight = 0

        # compute current path weight
        k = s
        for j in i:
            current_pathweight += graph[k][j]
            k = j
        current_pathweight += graph[k][s]

        # update minimum
        if current_pathweight < min_path:
            min_path = current_pathweight
            opt_path = i
        # min_path = min(min_path, current_pathweight)


    return min_path, opt_path


# Driver Code
if __name__ == "__main__":

    # matrix representation of graph
#     graph = [[0, 10, 15, 20], [10, 0, 35, 25],
#             [15, 35, 0, 30], [20, 25, 30, 0]]

    full_D = pd.read_csv('colleges.txt', sep=r"\s", index_col=0,
                        engine='python').to_numpy()
    
    # More than 10 locations is too many for my machine.
    V = 10
    graph = full_D[:V, :V]
    s = 0
    
    ans = travellingSalesmanProblem(graph, s)
    
    print(ans[0])
    print((s,) + ans[1] + (s,))
    
    
    
    
    
    
