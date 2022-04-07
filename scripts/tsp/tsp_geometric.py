# -*- coding: utf-8 -*-
"""
Algorithm from "THE TRAVELING SALESMAN PROBLEM", Thesis by Corinne Brucato,
University of Pittsburgh, 2013

A = set of vertices that lie on the hull
C = edges that lie on hull
T = set of vertices that do not lie on hull

1. Pick the vertex, v_i in T closest to the convex hull[a] and create all
     of the edges incident with v_i and the vertices in A (the hull set).

2. Measure all of the angles created this way. Let v_i, u_j and v_i, u_j+1 [b]
     be the edges that form the largest angle centered at v_i.
     Add these edges to C, and destroy the rest of the edges that are not in C
     along with the edge u_j, u_j+1 in C.

3.  Now, v_i is a member of A

4.  (Select the next vertex in T closes to the hull?) and repeat steps 1-3
     until T is empty

[a] closest euclidean distance to a hull edge
[b] u_j are vertices on the hull if we create edges in step 1 by following
the hull order then the orientation of the edges should increase monotonically.


"""
import math
from timeit import default_timer as timer
import pandas as pd
from tsp_evolutionary import College, route_plot


class Point:  # point class with x, y as point
    def __init__(self, x, y):
        self.x = x
        self.y = y


def Left_index(points):
    """
    Find the left most point
    """
    minn = 0
    for i in range(1, len(points)):
        if points[i].x < points[minn].x:
            minn = i
        elif points[i].x == points[minn].x:
            if points[i].y > points[minn].y:
                minn = i
    return minn


def orientation(p, q, r):
    '''
    To find orientation of ordered triplet (p, q, r).
    The function returns following values
    0 --> p, q and r are collinear
    1 --> Clockwise
    2 --> Counterclockwise
    '''
    val = (q.y - p.y) * (r.x - q.x) - \
        (q.x - p.x) * (r.y - q.y)

    if val == 0:
        return 0
    elif val > 0:
        return 1
    else:
        return 2


def convexHull(points, n):
    """Find convex hull of a set of points.
    https://www.geeksforgeeks.org/orientation-3-ordered-points/
    for explanation of orientation()
    This code is contributed by Akarsh Somani, IIIT Kalyani
    """

    # There must be at least 3 points
    if n < 3:
        return

    # Find the leftmost point
    l = Left_index(points)

    hull = []

    '''
    Start from leftmost point, keep moving counterclockwise
    until reach the start point again. This loop runs O(h)
    times where h is number of points in result or output.
    '''
    p = l
    q = 0
    while(True):

        # Add current point to result
        hull.append(p)

        '''
        Search for a point 'q' such that orientation(p, q,
        x) is counterclockwise for all points 'x'. The idea
        is to keep track of last visited most counterclock-
        wise point in q. If any point 'i' is more counterclock-
        wise than q, then update q.
        '''
        q = (p + 1) % n

        for i in range(n):

            # If i is more counterclockwise
            # than current q, then update q
            if(orientation(points[p], points[i], points[q]) == 2):
                q = i

        '''
        Now q is the most counterclockwise with respect to p
        Set p as q for next iteration, so that q is added to
        result 'hull'
        '''
        p = q

        # While we don't come to first point
        if(p == l):
            break

    return hull


def dist2line(s1, s2, p3):
    """
    Compute Euclidean distance to a line segment

    Parameters
    ----------
    s1 : tuple
        x, y coordinates of first point defining a line segment.
    s2 : tuple
        x, y coordinates of second point defining a line segment.
    p3 : tuple
        x, y coordinates of a point.

    Returns
    -------
    dist : float
        Euclidean (minimum) distance to line.

    """
    x1, y1 = s1[0], s1[1]
    x2, y2 = s2[0], s2[1]
    x3, y3 = p3[0], p3[1]

    px = x2-x1
    py = y2-y1

    norm = px*px + py*py

    u = ((x3 - x1) * px + (y3 - y1) * py) / float(norm)

    if u > 1:
        u = 1
    elif u < 0:
        u = 0

    x = x1 + u * px
    y = y1 + u * py

    dx = x - x3
    dy = y - y3

    """ Note: If the actual distance does not matter, if you only want to
    compare what this function returns to other results of this function,
    you can just return the squared distance instead (i.e. remove the sqrt)
    to gain a little performance"""

    dist = (dx*dx + dy*dy)**.5

    return dist


def dot(vA, vB):
    """Simple 2D dot product"""
    return vA[0]*vB[0]+vA[1]*vB[1]


def ang(lineA, lineB):
    """
    Computes the angle beween 2 lines defined by their endpoints.

    Parameters
    ----------
    lineA : tuple
        Two points defining a line. Each point is a tuple (x, y)
    lineB : tuple
        Two points defining a line. Each point is a tuple (x, y)

    Returns
    -------
    float
        angle between lineA and lineB.  Degrees

    """
    # Get nicer vector form
    vA = [(lineA[0][0]-lineA[1][0]), (lineA[0][1]-lineA[1][1])]
    vB = [(lineB[0][0]-lineB[1][0]), (lineB[0][1]-lineB[1][1])]
    # Get dot prod
    dot_prod = dot(vA, vB)
    # Get magnitudes
    magA = dot(vA, vA)**0.5
    magB = dot(vB, vB)**0.5
    # Get cosine value
    # cos_ = dot_prod/magA/magB
    # Get angle in radians and then convert to degrees
    angle = math.acos(dot_prod/magB/magA)
    # Basically doing angle <- angle mod 360
    ang_deg = math.degrees(angle) % 360

    if ang_deg - 180 >= 0:
        # As in if statement
        return 360 - ang_deg
    else:

        return ang_deg


def find_closest(C, T, colleges):
    """
    Fin the interior point that is closest to hulls edges.

    Parameters
    ----------
    C : list
        Hull edges.
    T : list
        Interior vertices.
    colleges : list
        list of college objects

    Returns
    -------
    vmin : int
        index of interior vertex closest to hull.

    """
    dmin = 1.e30
    for edge in C:
        s1 = (colleges[edge[0]].x, colleges[edge[0]].y)
        s2 = (colleges[edge[1]].x, colleges[edge[1]].y)

        for vertex in T:
            p3 = (colleges[vertex].x, colleges[vertex].y)
            d = dist2line(s1, s2, p3)
            if d < dmin:
                dmin = d
                vmin = vertex
    return vmin


def get_distance(route, dists):
    cost = 0
    if route[0] != route[-1]:
        route.append(route[0])

    for i in range(len(route)-1):
        cost += dists[route[i], route[i+1]]
    # cost += dists[route[-1], route[0]]
    return cost


def shift_route(route):
    """ Al Duke 3/24/2022
    Shifts sequence of cities so that the home city (index==0) is first.
    This makes comparisons with solutions from other algorithms easier since
    they maintain the home city as the origin.
    For example, if the home city ends up in 5th place on a route returned
    from a genetic algorithm, move the first 4 cities to the end of the
    sequence to create a "shifted" route.

    Parameters
    ----------
    route : list
        list of int.

    Returns
    -------
    shifted : list
        list of College object shifted so that College 0 (home) is first.

    """
    if route[-1] == route[0]:
        route = route[:-1]

    origin = 0
    for pos, val in enumerate(route):
        if val == 0:
            origin = pos
            break
    shifted = route[origin:] + route[:origin]

    return shifted


def geometric_tsp(colleges, dists):
    A = convexHull(colleges, len(colleges))
    # set of vertices that lie on the hull

    C = []  # edges that make up hull
    for i in range(len(A)-1):
        C.append((A[i], A[i+1]))
    C.append((A[-1], A[0]))

    T = [v for v in range(len(colleges)) if v not in A]  # vertices not on hull
    nsteps = len(T)

    while True:
        # find point closest to hull edges
        v_i = find_closest(C, T, colleges)

        # create a list of edges incident with v_i and the vertices in A
        temp_edges = []
        for v_h in A:
            temp_edges.append((v_i, v_h))

        # convert temp_edges into line objects that can be used by ang()
        # to compute included angles.
        lines = []
        for e in temp_edges:
            lines.append(((colleges[e[0]].x, colleges[e[0]].y),
                          (colleges[e[1]].x, colleges[e[1]].y)))
        lines.append(((colleges[temp_edges[-1][0]].x,
                       colleges[temp_edges[-1][0]].y),
                      (colleges[temp_edges[0][1]].x,
                       colleges[temp_edges[0][1]].y)))

        # calculate included angles between lines and find the largest.
        amax = 0
        for i in range(len(lines)-1):
            ai = ang(lines[i], lines[i+1])
            if ai > amax:
                amax = ai
                iline = i

        # edge1 and edge2 will be added to C.  edge3 will be removed
        # from the hull
        edge1 = temp_edges[iline]
        if iline == len(temp_edges)-1:
            edge2 = temp_edges[0]
        else:
            edge2 = temp_edges[iline+1]

        edge3 = (edge1[1], edge2[1])

        for x in set(C).intersection({edge3}):
            C.remove(x)

        # add edges connecting v_i to hull to C
        C.insert(iline, (edge1[1], edge1[0]))
        C.insert(iline+1, edge2)
        # add v_i to A
        A.insert(A.index(edge2[1]), v_i)
        # remove v_i from T
        for x in set(T).intersection({v_i}):
            T.remove(x)

        if len(T) == 0:
            break

    d = get_distance(A, dists)
    route = shift_route(A)
    route.append(0)
    return (d, route, nsteps)


def main():
    """."""

    dists = pd.read_csv('colleges.txt', sep=r"\s", index_col=0,
                        engine='python').to_numpy()

    # create colleges collection for list of colleges and associated towns
    df = pd.read_csv("college_locs.csv", index_col=0)

    colleges = []
    num_cities = len(df.index)
    for i in range(0, num_cities):
        colleges.append(College(df.iloc[i].long, df.iloc[i].lat,
                                df.iloc[i].college, i, dists))

    t0 = timer()
    ans = geometric_tsp(colleges, dists)
    tn = timer()
    route = [colleges[i] for i in ans[1]]  # route as list of colleges
    print('\nProcessing time: {:.3g} msec'.format((tn-t0)/1e-3))
    print("Starting at ", route[0].name)
    print('Path: ', ans[1])
    # print('Longest leg: ', longest_leg(dists, path_full))
    print('Geometric distance: ', ans[0])

    route_plot(route, "Geometric Solution", ans[0], ans[2])


if __name__ == "__main__":
    main()
