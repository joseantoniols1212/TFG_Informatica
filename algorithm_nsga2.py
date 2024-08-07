import numpy as np
import random
import array

from utils import get_data_by_index, \
    map_from_auxiliary_coordinates, map_to_auxiliary_coordinates

from deap import algorithms
from deap import base
from deap import creator
from deap import tools


def algorithm_nsga2(sigma, mu, esg,
                    penalization_weight, ngen):

    (n, m) = esg.shape
    mu = np.atleast_2d(mu).T
    centroid = np.random.rand(n)
    centroid = centroid / np.linalg.norm(centroid, 1)
    centroid = map_to_auxiliary_coordinates(centroid)
    centroid = np.append(centroid, np.random.rand())

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", array.array, typecode='d',
                   fitness=creator.FitnessMin)

    # We define evaluation matrix
    def eval_matrix(gamma):
        return gamma * sigma - mu@mu.T

    def penalization(ind):
        """ Penalizes portfolios not summing up to one."""
        w = ind
        norm = np.linalg.norm(w, 1)
        penalization = 0
        if norm > 1:
            penalization += penalization_weight * norm
        for x in w:
            if x < 0:
                penalization += penalization_weight * (-x)
        return penalization

    def evaluate(ind):
        w = ind[:-1]
        gamma = ind[-1]
        p = penalization(w)
        w = map_from_auxiliary_coordinates(w)
        w = np.atleast_2d(w).T
        h = eval_matrix(gamma)
        res = (w.T@h@w)[0]
        return abs(res) + p

    BOUND_LOW, BOUND_UP = 0.0, 1.0
    ETA = 20.0
    POPULATION_SIZE = 100
    CXPB = 0.9
    NDIM = n

    def uniform(low, up, size=None):
        try:
            return [random.uniform(a, b) for a, b in zip(low, up)]
        except TypeError:
            return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]

    toolbox = base.Toolbox()
    toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
    toolbox.register("individual", tools.initIterate,
                     creator.Individual, toolbox.attr_float)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxSimulatedBinaryBounded,
                     low=BOUND_LOW, up=BOUND_UP, eta=20.0)
    toolbox.register("mutate", tools.mutPolynomialBounded,
                     low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM)
    toolbox.register("select", tools.selNSGA2)

    pop = toolbox.population(n=POPULATION_SIZE)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # This is just to assign the crowding distance to the individuals
    # no actual selection is done
    pop = toolbox.select(pop, len(pop))

    # Begin the generational process
    for gen in range(1, ngen):

        # Vary the population
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= CXPB:
                toolbox.mate(ind1, ind2)

            toolbox.mutate(ind1)
            toolbox.mutate(ind2)
            del ind1.fitness.values, ind2.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Select the next generation population
        pop = toolbox.select(pop + offspring, POPULATION_SIZE)

    pop.sort(key=lambda x: x.fitness.values)
    w_p = pop[0][:-1]
    w_p = map_from_auxiliary_coordinates(w_p)
    gamma_0 = pop[0][-1]

    index_max = np.argmax(w_p.T@esg)

    w_s = np.zeros(m)
    w_s[index_max] = 1

    return w_p, w_s, gamma_0


if __name__ == "__main__":

    sigma, mu, esg = get_data_by_index(0)

    w_p, w_s, gamma_0 = algorithm_nsga2(sigma, mu, esg, 1000, 100)

    return_ = w_p.T@mu
    risk = w_p.T@sigma@w_p
    fitness = w_p.T@(gamma_0 * sigma - mu@mu.T)@w_p

    print("Portfolio optimo: ", w_p)
    print("Retorno: ", return_)
    print("Riesgo: ", risk)
    print("Norma portfolio:", np.linalg.norm(w_p, 1))
    print("Gamma_0: ", gamma_0)
    print("Fitness: ", fitness)
    print("Vector esg: ", w_s)
