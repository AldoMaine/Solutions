# -*- coding: utf-8 -*-
'''https://github.com/CarlEkerot/held-karp/blob/master/held-karp.py
'''
import itertools
import pandas as pd
from timeit import default_timer as timer


def held_karp(A):
    """
    Implementation of Held-Karp, an algorithm that solves the Traveling
    Salesman Problem using dynamic programming with memoization.
    Starting loc assumed to be 0.  Path omits return to 0 but net cost
    includes return to 0 cost.   From:
    https://github.com/CarlEkerot/held-karp/blob/master/held-karp.py

    Parameters:
        A: distance matrix

    Returns:
        A tuple, (cost, path).
    """
    n = len(A)

    # Maps each subset of the nodes to the cost to reach that subset, as well
    # as what node it passed before reaching this subset.
    # Node subsets are represented as set bits.
    costs = {}

    # Set transition cost from initial state
    for k in range(1, n):
        costs[(1 << k, k)] = (A[0][k], 0)

    # Iterate subsets of increasing length and store intermediate results
    # in classic dynamic programming manner
    for subset_size in range(2, n):
        for subset in itertools.combinations(range(1, n), subset_size):
            # Set bits for all nodes in this subset
            bits = 0
            for bit in subset:
                bits |= 1 << bit

            # Find the lowest cost to get to this subset
            for k in subset:
                prev = bits & ~(1 << k)

                res = []
                for m in subset:
                    if m in (0, k):
                        continue
                    res.append((costs[(prev, m)][0] + A[m][k], m))
                costs[(bits, k)] = min(res)

    # We're interested in all bits but the least significant (the start state)
    bits = (2**n - 1) - 1

    # Calculate optimal cost
    res = []
    for k in range(1, n):
        res.append((costs[(bits, k)][0] + A[k][0], k))
    opt, parent = min(res)

    # Backtrack to find full path
    route = []
    for _i in range(n - 1):
        route.append(parent)
        new_bits = bits & ~(1 << parent)
        _, parent = costs[(bits, parent)]
        bits = new_bits

    # Add implicit start state
    route.append(0)

    return opt, list(reversed(route))


def main():
    df_distances = pd.read_csv('colleges.txt', sep=r"\s", index_col=0,
                               engine='python')
    dists = df_distances.to_numpy()
    
    t0 = timer()
    ans = held_karp(dists)
    tn = timer()
    
    print(f'Processing time: {(tn-t0):.3g} sec'
          ' iterations.')
    print('Held-Karp distance: ', ans[0])
    print('Path: ', ans[1])


if __name__ == '__main__':
    main()
