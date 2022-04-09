"""Solution to the "Traveling Salesman" problem using a genetic algortihm.
Modified version of code found here
https://towardsdatascience.com/evolution-of-a-salesman-a-complete-genetic-algorithm-tutorial-for-python-6fe5d2b3ca35

"""
# import sys
import random
import operator
import time
from timeit import default_timer as timer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from progress.bar import IncrementalBar


class City:
    """City class."""

    def __init__(self, point, ref_idx):
        self.x = point[0]
        self.y = point[1]
        self.index = ref_idx

    def distance(self, city):
        """Distance."""
        x_distance = abs(self.x - city.x)
        y_distance = abs(self.y - city.y)
        distance = np.sqrt((x_distance ** 2) + (y_distance ** 2))
        return distance

    def __repr__(self):
        """."""
        return "(" + str(self.x) + "," + str(self.y) + ")"


class College:
    """College class."""

    def __init__(self, x, y, name, index, dists):
        self.x = x  # longitude
        self.y = y  # latitude
        self.name = name
        self.index = index
        self.dists = dists

    def distance(self, city):
        """Distance."""
        distance = self.dists[self.index, city.index]
        return distance

    def __repr__(self):
        """."""
        return "(" + self.name + " " + str(self.x) + "," + str(self.y) + ")"


class Fitness:
    """Define fitness as the inverse of the route distance.

    We want to minimize route distance, so a larger fitness score is better.
    """

    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness = 0.0

    def route_distance(self):
        """."""
        if self.distance == 0:
            path_distance = 0
            for j, city in enumerate(self.route):
                from_city = city
                to_city = None
                if j + 1 < len(self.route):
                    to_city = self.route[j + 1]
                else:  # travel back to starting point
                    to_city = self.route[0]
                path_distance += from_city.distance(to_city)
            self.distance = path_distance
        return self.distance

    def route_fitness(self):
        """."""
        if self.fitness == 0:
            self.fitness = 1 / float(self.route_distance())
        return self.fitness


def create_route(city_list):
    """Create an individual route.

    randomly select the order in which we visit the cities.

    returns a list of cities
    """
    route = random.sample(city_list, len(city_list))
    return route


def initial_population(pop_size, city_list):
    """Create a full population by looping through the create_route function.

    until we have as many routes as we want for our population.
    """
    population = []

    for _i in range(0, pop_size):
        population.append(create_route(city_list))

    return population


def rank_routes(population):
    """Use Fitness to rank each individual in the population.

    Returns ordered list with the route IDs and each associated fitness score.
    """
    fitness_results = {}
    for j, individual in enumerate(population):
        fitness_results[j] = Fitness(individual).route_fitness()
    return sorted(fitness_results.items(),
                  key=operator.itemgetter(1), reverse=True)


def selection(pop_ranked, elite_size):
    """Select the parents that will be used to create the next generation.

    Uses fitness proportionate selection (aka “roulette wheel selection”.
    The fitness of each individual relative to the population is used to
    assign a probability of selection. Think of this as the fitness-weighted
    probability of being selected.)
    """
    selection_results = []
    df_fit = pd.DataFrame(np.array(pop_ranked), columns=["Index", "Fitness"])
    df_fit['cum_sum'] = df_fit.Fitness.cumsum()
    df_fit['cum_perc'] = 100 * df_fit.cum_sum / df_fit.Fitness.sum()

    for j in range(0, elite_size):
        selection_results.append(pop_ranked[j][0])

    for _j in range(0, len(pop_ranked) - elite_size):
        pick = 100 * random.random()
        for k, candidate in enumerate(pop_ranked):
            if pick <= df_fit.iat[k, 3]:
                selection_results.append(candidate[0])
                break

    return selection_results


def mating_pool(population, selection_results):
    """Create the mating pool by extracting the selected individuals..

    ..from our population.
    """
    matingpool = []
    for _j, index in enumerate(selection_results):
        matingpool.append(population[index])

    return matingpool


def breed(parent1, parent2):
    """Create the next generation in a process called crossover.

    (aka “breeding” or "mating")
    This kind of crossover snips a segment out of P1 and places it
    at the beginning of child, the rest of child is taken in order
    from P2 being sure not to duplicate any genes.
    """
    child = []
    child_parent1 = []
    child_parent2 = []

    gene_a = int(random.random() * len(parent1))
    gene_b = int(random.random() * len(parent1))
    start_gene = min(gene_a, gene_b)
    end_gene = max(gene_a, gene_b)

    for j in range(start_gene, end_gene):
        child_parent1.append(parent1[j])

    child_parent2 = [item for item in parent2 if item not in child_parent1]

    child = child_parent1 + child_parent2
    return child


def breed_population(matingpool, elite_size):
    """Create offspring population."""
    children = []
    length = len(matingpool) - elite_size
    pool = random.sample(matingpool, len(matingpool))

    for j in range(0, elite_size):
        children.append(matingpool[j])

    for j in range(0, length):
        child = breed(pool[j], pool[len(matingpool) - j - 1])
        children.append(child)
    return children


def mutate(individual, mutation_rate):
    """Use swap mutation to swap the places of two cities in our route.

    ..for one individual
    """
    for swapped, _city in enumerate(individual):
        # if swapped == 0:  # Never swap out the first city
        #     continue
        if random.random() < mutation_rate:
            swap_with = int(random.random() * len(individual))
            city1 = individual[swapped]
            city2 = individual[swap_with]
            individual[swapped] = city2
            individual[swap_with] = city1

    return individual


# def mutate_population(population, mutation_rate):
#     """."""
#     mutated_population = []

#     for _j, individual in enumerate(population):
#         mutated_individual = mutate(individual, mutation_rate)
#         mutated_population.append(mutated_individual)

#     return mutated_population

def mutate_population(population, mutation_rate, elite_size):
    """."""
    mutated_population = []

    for _j, individual in enumerate(population):
        if _j > elite_size:
            mutated_individual = mutate(individual, mutation_rate)
            mutated_population.append(mutated_individual)
        else:  # Dont mutate elites
            mutated_population.append(individual)

    return mutated_population


def next_generation(current_generation, elite_size, mutation_rate):
    """."""
    pop_ranked = rank_routes(current_generation)
    selection_results = selection(pop_ranked, elite_size)
    matingpool = mating_pool(current_generation, selection_results)
    children = breed_population(matingpool, elite_size)
    next_gen = mutate_population(children, mutation_rate, elite_size)
    return next_gen


# def genetic_algorithm(population, pop_size, elite_size,
#                       mutation_rate, generations):
#     """."""
#     pop = initial_population(pop_size, population)
#     print("Initial distance: " + str(1 / rank_routes(pop)[0][1]))

#     starting_route = pop[rank_routes(pop)[0][0]]
#     # route_plot(starting_route, 'Starting route')
#     progress = [1e50]
#     route_plot(starting_route, 'Starting route (best of 100 random routes)',
#                progress[0], 1)  # distance, cycles

#     for _i in range(0, generations):
#         pop = next_generation(pop, elite_size, mutation_rate)

#     print("Final distance: " + str(1 / rank_routes(pop)[0][1]))
#     best_route_index = rank_routes(pop)[0][0]
#     best_route = pop[best_route_index]
#     # route_plot(best_route, 'Final route', 999, 1)
#     return (best_route, best_route, generations)


def genetic_algorithm(population, pop_size, elite_size, mutation_rate,
                      generations, stable=150):
    """

    Parameters
    ----------
    population : list
        List of places to visit.
    pop_size : int
        Number of individuals to generate for each new generation.
    elite_size : int
        Number of top ranked individuals to automatically add to breeding pool.
    mutation_rate : float
        The fraction of individuals in a new generation that has mutation.
    generations : int
        Limit on number of generations to spawn in search of the best soln.

    Returns
    -------
    global_best : tuple
        index and fitness of route with highest fitness (1/distance).
    global_route : list
        Ordered list of cities that form the shortest route.
    global_iter : int
        Number of generations required to first arrive at "best" solution.

    """

    plt.close('all')
    # generate pop_size (default=100) population by randomly sampling cities
    # from population with starting city fixed.
    pop = initial_population(pop_size, population)
    progress = []  # best route distance for each generation
    ranks = rank_routes(pop)
    global_best = ranks[0]  # tuple with (index of best frm pop, fitness score)
    global_route = pop[global_best[0]][:]
    global_iter = generations

    progress.append(1 / global_best[1])
    print((f"Shortest distance from {pop_size}"
           f" random routes: {1 / ranks[0][1]:.1f}"))

    no_improvement = 0  # counter for cycles without improvement
    for j in range(0, generations):
        pop = next_generation(pop, elite_size, mutation_rate)
        best_route = rank_routes(pop)[0]
        if best_route[1] > global_best[1]:  # found a better
            global_best = best_route[:]
            global_route = pop[global_best[0]][:]
            global_iter = j
            no_improvement = 0  # reset counter
        else:
            no_improvement += 1

        progress.append(1 / best_route[1])
        if no_improvement == stable:  # generations with no improvement
            break

    print(f"Best distance: {1 / global_best[1]:.1f}")

    home_route = shift_route(global_route)

    return (global_best, home_route, global_iter)


def genetic_algorithm_plot(population, pop_size, elite_size, mutation_rate,
                           generations, stable=150):
    """

    Parameters
    ----------
    population : list
        List of places to visit.
    pop_size : int
        Number of individuals to generate for each new generation.
    elite_size : int
        Number of top ranked individuals to automatically add to breeding pool.
    mutation_rate : float
        The fraction of individuals in a new generation that has mutation.
    generations : int
        Limit on number of generations to spawn in search of the best soln.

    Returns
    -------
    global_best : tuple
        index and fitness of route with highest fitness (1/distance).
    global_route : list
        Ordered list of cities that form the shortest route.
    global_iter : int
        Number of generations required to first arrive at "best" solution.

    """

    plt.close('all')
    # generate pop_size (default=100) population by randomly sampling cities
    # from population with starting city fixed.
    pop = initial_population(pop_size, population)
    progress = []  # best route distance for each generation
    ranks = rank_routes(pop)
    global_best = ranks[0]  # tuple with (index of best frm pop, fitness score)
    global_route = pop[global_best[0]][:]
    global_iter = generations

    progress.append(1 / global_best[1])
    print((f"Shortest distance from {pop_size}"
           f" random routes: {1 / ranks[0][1]:.1f}"))

    # progress indicator setup/start
    time.sleep(.2)  # avoid stdout race condition between print() and Bar
    bar = IncrementalBar('Generation', max=generations,
                         suffix='%(index)d/%(max)d')

    # starting_route = pop[global_best[0]]
    # route_plot(starting_route, 'Starting route (best of 100 random routes)',
    # progress[0], 1)

    no_improvement = 0  # counter for cycles without improvement
    for j in range(0, generations):
        pop = next_generation(pop, elite_size, mutation_rate)
        best_route = rank_routes(pop)[0]
        if best_route[1] > global_best[1]:  # found a better
            global_best = best_route[:]
            global_route = pop[global_best[0]][:]
            global_iter = j
            no_improvement = 0  # reset counter
        else:
            no_improvement += 1

        progress.append(1 / best_route[1])
        bar.next()
        if no_improvement == stable:  # generations with no improvement
            break

    bar._hidden_cursor = False
    bar.finish()

    # print(f"Best distance: {1 / global_best[1]:.1f}")

    # best_route = pop[best_route[0]]
    # route_plot(best_route, 'Final route', 1 / best_route[1], j + 1)

    route_plot(global_route, 'Best Route ', 1 / global_best[1], global_iter)

    plt.figure('Progression')
    plt.title('Shortest Route Distance vs. Generation')
    plt.plot(progress)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.show()

    home_route = shift_route(global_route)

    return (global_best, home_route, global_iter)


def route_plot(route, title, distance, cycles):
    """."""
    plt.figure()
    waypoints = np.empty((0, 2))
    for city in route:
        waypoints = np.append(waypoints, np.array([[city.x, city.y]]), axis=0)

    # add starting city to end of route to close the loop
    waypoints = np.append(waypoints, np.array([[route[0].x, route[0].y]]),
                          axis=0)

    try:
        for city in route:  # x,y in zip(xs,ys):
            plt.annotate(city.name,
                         (city.x, city.y),  # coordinates to position the label
                         textcoords="offset points",  # how to position text
                         xytext=(0, 6),  # distance from text to points (x,y)
                         ha='center')  # horizontal alignment
    except AttributeError:
        pass

    plt.plot(waypoints[:, 0], waypoints[:, 1], 'rs-')
    plt.title(f"{title}\nDistance: {distance:.1f}, Iterations: {cycles}")
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')


def loc_plot(route, title):
    """Plot college locations without route lines."""
    plt.figure(figsize=(7, 4.7))
    waypoints = np.empty((0, 2))
    for city in route:
        waypoints = np.append(waypoints, np.array([[city.x, city.y]]), axis=0)

    # add starting city to end of route to close the loop
    waypoints = np.append(waypoints, np.array([[route[0].x, route[0].y]]),
                          axis=0)

    try:
        for city in route:  # x,y in zip(xs,ys):
            plt.annotate(city.name,
                         (city.x, city.y),  # coordinates to position the label
                         textcoords="offset points",  # how to position text
                         xytext=(0, 6),  # distance from text to points (x,y)
                         ha='center')  # horizontal alignment
    except AttributeError:
        pass

    plt.scatter(waypoints[:, 0], waypoints[:, 1], )
    plt.title(f"{title}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")


def shift_route(route):
    """ Al Duke 3/15/2022
    Shifts sequence of cities so that the home city (index==0) is first.
    This makes comparisons with solutions from other algorithms easier since
    they maintain the home city as the origin.
    For example, if the home city ends up in 5th place on a route returned
    from a genetic algorithm, move the first 4 cities to the end of the
    sequence to create a "shifted" route.

    Parameters
    ----------
    route : list
        list of College objects.

    Returns
    -------
    shifted : list
        list of College object shifted so that College 0 (home) is first.

    """
    origin = 0
    for i, city in enumerate(route):
        if city.index == 0:
            origin = i
            break
    shifted = route[origin:] + route[:origin]

    return shifted


# %%
def main(city_list):
    """main."""

    t0 = timer()

    ans = genetic_algorithm_plot(population=city_list, pop_size=NUM_POP,
                                 elite_size=ELITES, mutation_rate=MUTATION,
                                 generations=MAX_GENERATIONS, stable=STABLE)
    tn = timer()

    loops = min(ans[2] + STABLE, MAX_GENERATIONS)
    
    print(f'Processing time: {(tn-t0):.3g} sec, {loops} iterations.')
    print('Generations to best route: ', ans[2])
    print('Best route: ', [c.index for c in ans[1]])
    print(f'Genetic Algo distance: {1/ans[0][1]:.1f}')


if __name__ == "__main__":
    # %% Simple test case
    # NUM_CITIES = 10  # number of cities to be visited
    # NUM_POP = 25  # number of candidate routes generated for each generation
    # ELITES = 5  # no. of fittest routes to auto add to breeding pool
    # MUTATE = .1  # mutation rate
    # MAX_GENERATIONS = 200  # limit on breeding cycles
    # STABLE = 20  # stop breeding after stable iterations with no improvement

    # random.seed(9)  # to get consistent results
    # city_list = []
    # for _i in range(NUM_CITIES):
    #     city_list.append(City((int(random.random() * 200),
    #                            int(random.random() * 200)), _i))

    # main(city_list)

    # %% 25 college tour
    df_distances = pd.read_csv('colleges.txt', sep=r"\s", index_col=0,
                               engine='python')
    D = df_distances.to_numpy()  # adjacency matrix

    df = pd.read_csv("college_locs.csv", index_col=0)

    colleges = []
    num_cities = len(df.index)
    for ii in range(num_cities):
        colleges.append(College(df.iloc[ii].long, df.iloc[ii].lat,
                                df.iloc[ii].college, ii, D))
        
    # loc_plot(colleges, "Colleges")

    NUM_POP = 100  # number of candidate routes generated for each generation
    MAX_GENERATIONS = 500  # limit on mating cycles
    ELITES = 10  # auto included in breeding pool, not mutated
    STABLE = 150
    MUTATION = .015

    main(colleges)
