import numpy as np
import random
import array

from utils import get_data_by_index, \
    map_from_auxiliary_coordinates, map_to_auxiliary_coordinates

from deap import base
from deap import creator
from deap import tools

from metrics import *

creator.create("FitnessMaxNSGA2", base.Fitness, weights=(1.0,))
creator.create("IndividualMaxNSGA2", array.array, typecode='d',
               fitness=creator.FitnessMaxNSGA2)


def algorithm_nsga2(sigma, mu, esg, pop_size, penalization_weight, ngen):

    (n, m) = esg.shape
    mu = np.atleast_2d(mu).T

    def penalization(ind):
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
        w = ind
        p = penalization(w)
        w = map_from_auxiliary_coordinates(w)
        w = np.atleast_2d(w).T
        res = (w.T@mu / (w.T@sigma@w)**0.5)[0]
        return res - p

    BOUND_LOW, BOUND_UP = 0., 1.
    ETA = 20.
    POPULATION_SIZE = pop_size
    CXPB = 0.9
    NDIM = n-1

    def uniform(low, up, size=None):
        try:
            return [random.uniform(a, b) for a, b in zip(low, up)]
        except TypeError:
            return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]

    toolbox = base.Toolbox()
    toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
    toolbox.register("individual", tools.initIterate,
                     creator.IndividualMaxNSGA2, toolbox.attr_float)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxSimulatedBinaryBounded,
                     low=BOUND_LOW, up=BOUND_UP, eta=ETA)
    toolbox.register("mutate", tools.mutPolynomialBounded,
                     low=BOUND_LOW, up=BOUND_UP, eta=ETA, indpb=1.0/NDIM)
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
    w_p = pop[0]
    w_p = map_from_auxiliary_coordinates(w_p)

    index_max = np.argmax(w_p.T@esg)

    w_s = np.zeros(m)
    w_s[index_max] = 1

    return w_p, w_s


if __name__ == "__main__":
    POPULATION_SIZE = 100
    NGEN = 100

    sigma, mu, esg = get_data_by_index(0)

    assets = Assets(sigma, mu, esg)

    w_p, w_s = algorithm_nsga2(sigma, mu, esg, POPULATION_SIZE, 1000, NGEN)

    portfolio = Portfolio(w_p, w_s)
    metrics = Metrics(portfolio, assets)

    metrics.report()
